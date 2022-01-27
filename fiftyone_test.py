import fiftyone as fo


dataset_dir = "./dataset/training"


name = "test_dataset1"


dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir, dataset_type=fo.types.KITTIDetectionDataset, name=name
)
sess = fo.launch_app(dataset)
sess.wait()
