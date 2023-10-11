import os
import shutil
import uuid

try:
    os.mkdir('./cgan_ready')
except:
    pass

# Get all the folders here
folders = [x for x in os.listdir('./') if os.path.isdir(x) and not 'cgan_ready' in x]

for folder in folders:

    # Prepare the final directory
    try:
        os.mkdir(os.path.join('./cgan_ready', folder))
    except:
        pass
    try:
        os.mkdir(os.path.join('./cgan_ready', folder, 'normal'))
    except:
        pass
    try:
        os.mkdir(os.path.join('./cgan_ready', folder, 'abnormal'))
    except:
        pass

    # Train subfolders
    subdirs = [os.path.join('./', folder, 'train', x) for x in os.listdir(os.path.join('./', folder, 'train')) if os.path.isdir(os.path.join('./', folder, 'train', x))]
    
    n = 0

    # Copy the images to the normal folder
    for subdir in subdirs:
        for image in os.listdir(subdir):
            if 'good' in subdir:
                shutil.copy(os.path.join(subdir, image), os.path.join('./cgan_ready', folder, 'normal', str(n)+'.png'))
            else:
                shutil.copy(os.path.join(subdir, image), os.path.join('./cgan_ready', folder, 'abnormal', str(n)+'.png'))
            n += 1

    # Test subfolders
    subdirs = [os.path.join('./', folder, 'test', x) for x in os.listdir(os.path.join('./', folder, 'test')) if os.path.isdir(os.path.join('./', folder, 'test', x))]

    # Copy the images to the abnormal folder
    for subdir in subdirs:
        for image in os.listdir(subdir):
            if 'good' in subdir:
                shutil.copy(os.path.join(subdir, image), os.path.join('./cgan_ready', folder, 'normal', str(n)+'.png'))
            else:
                shutil.copy(os.path.join(subdir, image), os.path.join('./cgan_ready', folder, 'abnormal', str(n)+'.png'))
            n += 1