import glob


class Config:
    artifact_path =  "./artfact/exp-test"
    #  Transformer Encoder parameters
    encoder_model_name = ["csebuetnlp/banglabert", "sagorsarker/bangla-bert-base"][0]  # change the model name if you want to use a different encoder
    max_seq_length = 64  # tokenizer max length; could be up to 512
    train_batch_size = 64
    val_batch_size = 64
    input_size = 768  # same as encoder_model output size
    hidden_size = 768  # hidden, output size of discriminator
    #  GAN-BERT specific parameters
    generator_noise_size = 100  # size of the output space
    num_hidden_layers_g = 1  # number of hidden layers in the generator
    num_hidden_layers_d = 1  # number of hidden layers in the discriminator
    out_dropout_rate = 0.2  # dropout to be applied to discriminator's input vectors
    #  Optimization parameters
    epsilon = 1e-8
    num_train_epochs = 50
    learning_rate_discriminator = 5e-5
    learning_rate_generator = 5e-5
    # Scheduler
    apply_scheduler = False
    warmup_proportion = 0.1
    # dataset
    labeled_file = "./data/labeled.tsv"
    unlabeled_file = "./data/unlabeled.tsv"
    test_filename = "./data/test.tsv"
    apply_balance = True  # Replicate labeled data to balance poorly represented datasets


    files_path = glob.glob('./dataset/BNIntent30/*.json')
    label_list = """calendar
                date
                thank_you
                time
                next_song
                goodbye
                how_old_are_you
                change_user_name
                change_volume
                tell_joke
                current_location
                todo_list
                fun_fact
                traffic
                distance
                where_are_you_from
                translate
                calculator
                weather
                spelling
                reminder
                make_call
                alarm
                what_are_your_hobbies
                no
                play_music
                repeat
                yes
                what_is_your_name
                what_song""".split('\n')

    label_list = [i.strip() for i in label_list]
    label2class = {cls: idx for idx, cls in enumerate(label_list)}
