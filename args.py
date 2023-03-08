class args():
    epochs = 16
    batch_size = 4

    datasets_prepared = False

    train_ir = 'E:\clip picture last 128/ir'
    train_vi = 'E:\clip picture last 128/vi'


    hight = 128
    width = 128
    image_size = 128

    save_model_dir = "models_training_5"
    save_loss_dir = "loss"

    cuda = 1

    g_lr = 0.0001
    d_lr = 0.0004
    log_interval = 5
    log_iter = 1