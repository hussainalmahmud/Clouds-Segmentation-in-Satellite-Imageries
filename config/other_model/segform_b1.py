config = dict(
    encoder='nvidia/segformer-b1-finetuned-ade-512-512',  # This is the encoder you mentioned earlier, adjust as needed.
    model_network='AmpNet_2',
    in_channels=3,
    n_class=1,
    save_path='',
    learning_rate=0.0005,
    weight_decay=0.0005,
    seed=10000,
)
config['save_path']='{}_{}'.format(config['model_network'],config['encoder'])
