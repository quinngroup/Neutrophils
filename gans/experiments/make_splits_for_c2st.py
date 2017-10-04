import os
import random

classes_of_interest = ['neutrophils']

data_root = '/media/narita/Data/Neutrophils/gan_images'
data_paths = ['neutrophil_train',
              'neutrophil_test']

target_folders = ['neutrophils_train_splits',
                  'neutrophils_test_splits']

for data_path, target_folder in zip(data_paths, target_folders):
    data_path = os.path.join(data_root, data_path)
    target_folder = os.path.join(data_root, target_folder)
    print('Working with', data_path)

    os.mkdir(target_folder)

    # split number, split role (train or test), class_id
    split_path = os.path.join(target_folder, 'split{0}', '{1}', '{2}')

    random_seed = 1
    num_splits = 10
    num_classes = len(classes_of_interest)

    for class_id in classes_of_interest:
        for i_split in range(num_splits):
            print('Class {0}, split {1}'.format(class_id, i_split))

            random.seed(i_split + random_seed)

            #Read all images from the data_path folder for current class
            class_path = os.path.join(data_path, class_id)
            print ('Class path: ' +class_path)
            class_images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

            #Sort those images
            class_images.sort()

            #Shuffle those images
            random.shuffle(class_images)

            test_path = split_path.format(i_split, 'test', class_id)
            train_path = split_path.format(i_split, 'train', class_id)
            print ('Test path: '+test_path)
            print ('Train path: '+train_path)
            print (' ')

            os.makedirs(test_path)
            os.makedirs(train_path)

            num_test = len(class_images) // 2
            for i_im in range(num_test):
                os.system('ln -rs {0} {1}'.format(os.path.join(class_path, class_images[i_im]),
                                                  os.path.join(test_path, class_images[i_im])))

            for i_im in range(num_test, len(class_images)):
                os.system('ln -rs {0} {1}'.format(os.path.join(class_path, class_images[i_im]),
                                                  os.path.join(train_path, class_images[i_im])))
