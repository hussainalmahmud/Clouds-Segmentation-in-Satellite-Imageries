config = {'encoder': 'none', 
          'model_network': 'refined_unet', 
          'in_channels': 3, 
          'n_classes': 1, 
          'save_path': '', 
          'learning_rate': 0.0003, 'weight_decay': 0.0005, 'seed': 10000}
config['save_path']='{}_{}'.format(config['encoder'],config['model_network'])