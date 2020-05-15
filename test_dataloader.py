



if __name__ == "__main__":
    # dataset
    train_dataloader = getDataLoader(
        args.dataset, args.batch_size, args.dataset_path, 'train', shuffle=True, augment=True)
    