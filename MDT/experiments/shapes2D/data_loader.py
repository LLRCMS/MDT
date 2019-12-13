import numpy as np
import os
from collections import OrderedDict
import time
import subprocess

# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
# from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates
from batchgenerators.transforms.utility_transforms import NullOperation

from shapes import ShapesDataset, extract_bboxes

def get_train_generators(cf, logger):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators.
    selects patients according to cv folds (generated by first run/fold of experiment):
    splits the data into n-folds, where 1 split is used for val, 1 split for testing and the rest for training. (inner loop test set)
    If cf.hold_out_test_set is True, adds the test split to the training data.
    """
    all_data = load_dataset(cf, logger)

    batch_gen = {}

    batch_gen['train'] = create_data_gen_pipeline(
        all_data, cf=cf, nbrSamples =  cf.num_train_batches * cf.batch_size, do_aug=False)
    batch_gen['val_sampling'] = create_data_gen_pipeline(
        all_data, cf=cf, nbrSamples = cf.num_val_batches, do_aug=False)

    batch_gen['n_val'] = cf.num_val_batches

    return batch_gen


def get_test_generator(cf, logger):
    """
    wrapper function for creating the test batch generator pipeline.
    selects patients according to cv folds (generated by first run/fold of experiment)
    If cf.hold_out_test_set is True, gets the data from an external folder instead.
    """
    print("Warning: 'get_test_generator' not implemented")

    return


def load_dataset(cf, logger, subset_ixs=None):
    """
    loads the dataset.
    :param subset_ixs: subset indices to be loaded from the dataset. used e.g. for testing to only load the test folds.
    :return: data: dictionary with one entry per sample. each entry is a dictionary containing respective meta-info as
    well as paths to the preprocessed numpy arrays to be loaded during batch-generation
    """
    if cf.debug_data_loader:
        print(">>> load_dataset")
        print ("  cf.pp_data_path, cf.input_df_name",
                   cf.pp_data_path, cf.input_df_name )

    if cf.server_env:
        print("Error: mode not available")
        exit()

    # Create Shapes Dataset
    #from shapes import ShapesDataset, extract_bboxes, non_max_suppression
    dataset = ShapesDataset(num_samples=cf.num_samples, height= cf.pre_crop_size_2D[0], width=cf.pre_crop_size_2D[1])

    dataset.load_shapes()
    dataset.prepare()

    # p_df = pd.read_pickle(os.path.join(cf.pp_data_path, cf.input_df_name))
    if subset_ixs is not None:
        # subset_pids = [np.unique(p_df.pid.tolist())[ix] for ix in subset_ixs]
        # p_df = p_df[p_df.pid.isin(subset_pids)]
        # logger.info('subset: selected {} instances from df'.format(len(p_df)))
        subset_pids = [np.unique(dataset.image_ids)[ix] for ix in subset_ixs]
        logger.info('subset: selected {} instances from df'.format(len(subset_pids)))

    pids = list(dataset.image_ids)

    if (len(pids) < cf.batch_size * cf.num_train_batches):
        logger.info(
        "Warning: number of read events not sufficient {} read / {} required"
        .format(len(pids),  cf.batch_size * cf.num_train_batches ) )

    data = OrderedDict()

    # SD - modified based on shapes dataset, need to fix
    labelIDs = []
    for ix, pid in enumerate(pids):
        _, class_id = dataset.load_mask(pid)
        labelIDs.append( list(class_id) )

    data['dataset'] = dataset
    data['pids'] = pids # TODO: This is unused. Modify for use or remove.
    data['labels'] = labelIDs
    data['xSize'] = dataset.width
    data['ySize'] = dataset.height

    return data


def create_data_gen_pipeline(sample_data, cf, nbrSamples=0, do_aug=True):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param sample_data: dictionary containing one dictionary per sample in the train/test subset.
    :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """

    # create instance of batch generator as first element in pipeline.
    data_gen = BatchGenerator(sample_data, batch_size=cf.batch_size, nbrSamples=nbrSamples,
                              cf=cf)

    # add transformations to pipeline.
    my_transforms = []
    if do_aug:
        mirror_transform = Mirror(axes=np.arange(2, cf.dim+2, 1))
        my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'])

        my_transforms.append(spatial_transform)
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    my_transforms.append(NullOperation(cf.dim))
    # GG TODO : test if "my_transforms = []" is sufficient
    my_transforms = []

    all_transforms = Compose(my_transforms)

    multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
    # multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
    return multithreaded_generator


############################################################
#  Pytorch Batch Generator
############################################################

class BatchGenerator(SlimDataLoaderBase):
    """
    creates the training/validation batch generator. Samples n_batch_size items (draws a slice from each item if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of items to sample for the batch
    :return dictionary containing the batch data (b, c, x, y, (z)) / seg (b, 1, x, y, (z)) / pids / class_target
    """
    def __init__(self, data, batch_size, nbrSamples, cf):
        # Nbr of event in the data-set
        super(BatchGenerator, self).__init__(data, batch_size)

        self.cf = cf
        self.nbrOfBatchProcessed = 0 # Nbr of batch already processed
        self.shuffleEvIdx = np.zeros( nbrSamples, dtype=np.int32)
        self.nbrOfSamples = nbrSamples
        self.nbrOfBatch = int( nbrSamples  / batch_size )

    def generate_train_batch(self):
        """
        return values required by MRCNN model
        'data'       batch_images     np.shape(B, C, xSize, ySize)
        'roi_labels' batch_gt_classes [ B, np.shape(nObjs) ],  int
        'bb_target'  batch_gt_bboxes  [ B, np.shape(nObjs, 4) ] ~ [B, nObjs, (x1, y1, x2, y2)]
        'roi_masks'  batch_gt_masks   [ B, np.shape(nObjs, C, xSize, ySize) ]

        For post processing
        'pid'        batch_pid        [ bs ]

        With
        B : batch size
        C : number of image channels
        xSize, ySize : image size
        nObjs : number of objects in the images
        """

        if self.cf.debug_generate_train_batch:
            print(">> generate_train_batch self._data", self.__dict__['_data'].keys() )

        # Shuffle at each epoch
        if ( self.nbrOfBatchProcessed == 0):
            self.shuffleEvIdx = np.array(
                [i for i in range(self.nbrOfSamples) ], dtype=np.int32)
            if not self.cf.debug_deactivate_shuffling:
                np.random.shuffle( self.shuffleEvIdx )
                print("  Shuffling ... ");


        dataset = self._data['dataset']
        pids = self._data['pids'] # TODO: This is unused. Modify for use or remove.
        labelsIDs = self._data['labels']

        if self.cf.debug_generate_train_batch :
            print("#images, #labels", len(pids), len(labelIDs))

        B = self.batch_size
        C = self.cf.n_channels
        xSize = self._data['xSize']
        ySize = self._data['ySize']

        # Allocate/type return values
        batch_images     = np.zeros( (B, C, xSize, ySize) )
        batch_pid        = [] # Image index (used in post-processing)
        batch_gt_classes = []
        batch_gt_bboxes  = []
        batch_gt_masks   = []

        for ii in range(self.batch_size) :
            idx = self.batch_size * self.nbrOfBatchProcessed + ii

            # Debug
            if self.cf.debug_generate_train_batch :
                print("  idx, self.shuffleEvIdx[ ii ]", idx, self.shuffleEvIdx[ ii ])
            evIdx = self.shuffleEvIdx[ idx ]

            # Load GT images, masks, class labels/IDs, bounding boxes using evIdx
            image = dataset.load_image(evIdx) # orig shape is [H,W,C]
            mask, class_label = dataset.load_mask(evIdx) # orig mask shape is [H,W,nObjs]
            class_id = dataset.map_classnames_to_classids(class_label) # this is a numpy array
            bbox = extract_bboxes(mask) # orig shape is [nObjs,(y1,x1,y2,x2)] ~ [nObjs,4]
             
            # Expands dims of class_id
            class_id = np.expand_dims(class_id, axis=-1) 

            # Images
            # NOTE: take care to transpose correctly!
            batch_images[ii,:,:,:] = np.transpose(image, (2,1,0)) # shape is now [B,C,W,H]

            # GT classes
            nObjs = len( labelsIDs[evIdx] )
            if self.cf.debug_generate_train_batch:
                print("number of objects is ", nObjs)

            # BBoxes and Masks allocation
            gt_bboxes = np.zeros( (nObjs, 4), dtype=np.float )
            gt_masks  = np.zeros( (nObjs, C, xSize, ySize) )
           
            for k in range(nObjs):
                # BBoxes
                y1, x1, y2, x2 = bbox[k,:]
                gt_bboxes[k, :] = x1, y1, x2, y2 # format is now [nObjs, (x1,y1,x2,y2)]

                # Masks
                # NOTE: take care to transpose and expand dimension correctly!
                mask_k = np.expand_dims(mask[:,:,k], axis=-1) # shape is now [H,W,1]
                gt_masks[k,0,:,:] =  np.transpose(mask_k, (2,1,0)) # shape is now [nObjs,C,W,H]

            # Batch update
            batch_pid.append( evIdx )
            batch_gt_classes.append( class_id )
            batch_gt_masks.append( gt_masks )
            batch_gt_bboxes.append( gt_bboxes )

        # Update values which control the shuffling
        self.nbrOfBatchProcessed += self.batch_size
        if( self.nbrOfBatchProcessed >= self.nbrOfBatch): self.nbrOfBatchProcessed = 0
        return { 'data': batch_images, 'pid': batch_pid, 'roi_masks': batch_gt_masks,
                 'roi_labels': batch_gt_classes, 'bb_target': batch_gt_bboxes }

