import neptune
import numpy as np

PARAMS = {'n_iterations': 117,
          'n_images': 5}

neptune.init('davidturner94/sandbox',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZDA3MmRiNzItNDhjZS00MWM5LThiYmUtYTc4M2ZiYWExODQyIn0=')

with neptune.create_experiment(name='start-with-neptune',
                               params=PARAMS):

    neptune.append_tag('first-example')

    for i in range(1, PARAMS['n_iterations']):
        neptune.log_metric('iteration', i)
        neptune.log_metric('loss', 1/i**0.5)
        neptune.log_text('magic values', 'magic value {}'.format(0.95*i**2))

    for j in range(0, PARAMS['n_images']):
        array = np.random.rand(10, 10, 3)*255
        array = np.repeat(array, 30, 0)
        array = np.repeat(array, 30, 1)
        neptune.log_image('mosaics', array)
