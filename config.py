

class Config:
    artifact_path = "./artifact/exp-1"
    #  Transformer Encoder parameters
    encoder_model_name = "bert-base-cased"  # change the model name if you want to use a different encoder
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
    num_train_epochs = 10
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

    label_list = [
        "UNK_UNK","ABBR_abb", "ABBR_exp", "DESC_def", "DESC_desc",
        "DESC_manner", "DESC_reason", "ENTY_animal", "ENTY_body",
        "ENTY_color", "ENTY_cremat", "ENTY_currency", "ENTY_dismed",
        "ENTY_event", "ENTY_food", "ENTY_instru", "ENTY_lang",
        "ENTY_letter", "ENTY_other", "ENTY_plant", "ENTY_product",
        "ENTY_religion", "ENTY_sport", "ENTY_substance", "ENTY_symbol",
        "ENTY_techmeth", "ENTY_termeq", "ENTY_veh", "ENTY_word", "HUM_desc",
        "HUM_gr", "HUM_ind", "HUM_title", "LOC_city", "LOC_country",
        "LOC_mount", "LOC_other", "LOC_state", "NUM_code", "NUM_count",
        "NUM_date", "NUM_dist", "NUM_money", "NUM_ord", "NUM_other",
        "NUM_perc", "NUM_period", "NUM_speed", "NUM_temp", "NUM_volsize",
        "NUM_weight"
    ]
    label2class = {cls: idx for idx, cls in enumerate(label_list)}
