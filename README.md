# cnn-lstm-gan-music-generation

EDIT: An outline for this GAN scheme is added in my [Bachelor thesis](thesis.pdf) , which experiments with LSTM skip-connections for music generation with very robust evaluation.

This program is used to enhance the outputs of recurrent neural music generator, it could also be used to tune human composed music.


Work in progress, repository doesn't contain midis, or needed files, directories or weights. Just up to date code as my backup.

![how it works](https://cloud.githubusercontent.com/assets/13591225/25017098/c981fd9e-2082-11e7-8574-aaea5a4174bc.gif)

One example of enhancements can be found in discriminator_memory folder.

It takes music from this network(in my MusicNetwork repo):

![first model](https://cloud.githubusercontent.com/assets/13591225/25025151/6071e26e-20a1-11e7-870d-25f623b627b8.png)


The proposed model diagram, (more detailed in code, there might be mistakes in the picture):

![EGAN](https://cloud.githubusercontent.com/assets/13591225/25025240/b8df0620-20a1-11e7-9e9c-f45dd9c91e19.png)

