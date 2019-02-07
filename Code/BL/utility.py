import pandas as pd
import numpy as np
import h5py

from keras import backend as K
import tensorflow as tf




def adjust_prob(prob, b = 0.5, b_prime = 0.001):
        """The formula is
                p' = b'*(p - pb)/(b - pb + b'p - bb')
                b: base rate
                b_prime: base rate in population
                prob: probability in model
                prob_prime: adjusted probability based on bayesian minimum risk
        """
	p_prime = b_prime * (prob - prob * b)/(b - prob * b + b_prime * prob - b * b_prime)
        return p_prime


def read_hdf5(file_name, kwargs):
        import numpy as np
        data_block = h5py.File(file_name)
        rslt = []
        for arg in kwargs:
                print arg
                var = arg + '= np.array(data_block[\'' + arg + '\']);' + 'rslt.append(' + arg + ')'
                print var
                exec (var, locals())
        data_block.close()
        return tuple(rslt)


def save_hdf5(file_name, args1, args2):
    h5f = h5py.File(file_name, 'w')
    if len(args1) != len(args2):
        print('length does not math!')
    for i in range(len(args1)):
        key = args1[i]
        print key
        var = key + ' = args2[i];' + 'h5f.create_dataset(\'' + key + '\'' + ', ' + \
                'data=' + key + ', ' + 'compression=' + '\'gzip\'' + ', ' + 'compression_opts=9)'
        exec (var, locals())
    h5f.close()


def print_id_label(pname, np_guid, np_churn):
    """print the number of unique users and
    churn labels
    """
    print(pname + ':****************')
    print(len(set(np_guid)))
    print('churn label count', np.unique(np_churn, return_counts=True))





#input_keys = ['msno', 'event' ,'m_y', 'mask', 'y', 'bd', 'registration_init_time', 'cgr', 'is_auto_renew', 'is_cancel']


def approx_gradient(model):
    """approximate the gradient based on
    the first-order Taylor expansion
    Simonyan et al., Deep inside convolutional networks: Visualising image
    classification models and saliency maps. arXiv preprint arXiv:1312.6034 (2013).
    """
    #use the last but one layer for better performance
    main_X = model.get_layer('main_input').output
    aux_s_X = model.get_layer('s_input').output
    cond_state = model.get_layer('status_input').output
    aux_d_X = model.get_layer('d_input').output
    y = model.layers[-1].output[:,-1,0]
    #print(main_X._keras_shape, aux_X._keras_shape, cond_state._keras_shape, y._keras_shape)
    gradients = K.gradients(y, [main_X, aux_s_X, cond_state, aux_d_X])
    """sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    #get gradients in test mode
    saliency = sess.run(gradients, feed_dict={main_X: main_samples, aux_s_X: aux_s_samples, \
			cond_state: cond_state_samples, aux_d_X: aux_d_samples, K.learning_phase(): 0})
    sess.close()"""
    return K.function([main_X, aux_s_X, cond_state, aux_d_X, K.learning_phase()], gradients)


def get_saliency(model, main_X, aux_s_X, cond_stat_test, aux_d_X, idxs):
        for i in range(len(idxs)):
                print(i)
                idx = idxs[i]
                main_input = main_X[idx]
                aux_s_input = aux_s_X[idx]
                cond_input = cond_stat_test[idx]
                aux_d_input = aux_d_X[idx]		
                smap_i = approx_gradient(model)([main_input, aux_s_input, cond_input, aux_d_input, 0])
                print('done')
                if i == 0:
                    main_smap = np.sum(np.absolute(smap_i[0]), axis = 0)
                    aux_s_smap = np.sum(np.absolute(smap_i[1]), axis = 0)
                    cond_smap = np.sum(np.absolute(smap_i[2]), axis = 0)
                    aux_d_smap = np.sum(np.absolute(smap_i[3]), axis = 0)
                else:
                    main_smap_i = np.sum(np.absolute(smap_i[0]), axis = 0)
                    aux_s_smap_i = np.sum(np.absolute(smap_i[1]), axis = 0)
                    cond_smap_i = np.sum(np.absolute(smap_i[2]), axis = 0)
                    aux_d_smap_i = np.sum(np.absolute(smap_i[3]), axis = 0)
                    main_smap = np.add(main_smap, main_smap_i)
                    aux_s_smap = np.add(aux_s_smap, aux_s_smap_i)
                    cond_smap = np.add(cond_smap, cond_smap_i)
                    aux_d_smap = np.add(aux_d_smap, aux_d_smap_i)
        n = main_X.shape[0]
        main_smap = main_smap/n
        aux_s_smap = aux_s_smap/n
        cond_smap = cond_smap/n
        aux_d_smap = aux_d_smap/n
        smap = [main_smap, aux_s_smap, cond_smap, aux_d_smap]
        return smap

