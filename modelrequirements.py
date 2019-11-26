class ModelRequirements:
    SOURCE_IMAGE_PATH ="C:\\Users\Yuliya_Harshkova\\PycharmProjects\\first_work\\sources\\input\\"
    UPSCALED_IMAGE_PATH ="C:\\Users\Yuliya_Harshkova\\PycharmProjects\\first_work\\sources\\output\\"
    TFRECORDS_PATH = "C:\\Users\Yuliya_Harshkova\\PycharmProjects\\first_work\\tfrecords\\"
    LIST_OF_SCALE_FACTORS = [2, 4, 8, 16, 32]
    LEARNING_RATE = 0.0025
    BATCH_SIZE_ = 2
    EPOCH_AMOUNT = 1000
    EPOCH_BORDER = 250
    MODEL_OPTIMIZER = ""
    MODEL_ACTIVATION_FUNCTION = ""
    PERCENT_OF_VALIDATION_IMAGES = 30
    CROP_HEIGHT = 300
    CROP_WIGHTS = 400
    COLORSPACE = 'RGB'
    CHANNELS = 3
    IMAGE_FORMAT = 'JPEG'
    #NUMBER OF THREADS FOR TFRECORDS
    RANGES_FOR_RECORDS = 2
    RECORDS_BATCH_SIZE = 500
