# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 14:22:13 2018

@author: wmy
"""

'''
Google Inception Net V3
Deepth: 42 layers
'''

import tensorflow as tf
import time
import math
from datetime import datetime 

'''
为何选择TF-Slim:
    TF-Slim是一个简化构建，训练和评估神经网络的库：
        允许用户通过消除样板代码来更紧凑地定义模型。
        这是通过使用参数范围和许多高级层和变量来实现的。
        这些工具提高了可读性和可维护性，降低了复制和粘贴超参数值的错误发生的可能性，
        并简化了超参数调整。
        通过提供常用的正则化器使开发模型变得简单。
        几种广泛使用的计算机视觉模型（例如，VGG，AlexNet）已经开发出来，
        并且可供用户使用。这些可以用作黑盒子，或者可以以各种方式扩展，
        例如，通过向不同的内部层添加“多个头”。
        Slim可以轻松扩展复杂模型，
        并通过使用预先存在的模型检查点来热启动训练算法。
'''

# 导入tensorflow.contrib.slim模块
slim = tf.contrib.slim

# 定义初始化函数trunc_normal, 参数：随机数均值0.0, 标准差stddev
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# 定义一个超参数作用域函数
def inception_v3_arg_scope(weight_decay=0.00004,
                           stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
    '''
    Args:
        weight_decay: l2正则化的weight_decay参数
        stddev: 初始化标准差
        batch_norm_var_collection： 默认'moving_vars'
    '''
    
    # 批量归一化的参数
    batch_norm_params = {
        # 衰减系数
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],  
        }
    }
        
    # 对于卷积层和全连接层，使用l2正则化
    with slim.arg_scope([slim.conv2d,slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay)):
        # 对于卷积网络，设定权重初始化方法，激活函数relu, 批量归一化为归一化方法
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params) as sc:
            
            # 返回scope
            return sc
        
def inception_v3_base(inputs, scope=None):
    '''
    Args:
        inputs: image dataset tensor
        scope: setted scope
    '''
    
    # 用于保存关键节点
    end_points = {}
    
    # 网络搭建
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        # 网络前半部分
        
        # 默认：对于卷积层， 最大池化层， 平均池化层，卷积步长为1， 无padding
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
            # layer 1 input 299*299*3 output 149*149*32
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope="Conv2d_1a_3x3")
            # layer 2 input 149*149*32 output 147*147*32
            net = slim.conv2d(net, 32, [3, 3], scope="Conv2d_2a_3x3")
            # layer 3 input 147*147*32 output 147*147*64
            net = slim.conv2d(net, 64, [3, 3], padding="SAME",
                              scope="Conv2d_2b_3x3")
            # layer 4 input 147*147*64 output 73*73*64
            net = slim.max_pool2d(net, [3, 3], stride=2, scope="MaxPool_3a_3x3")
            # layer 5 input 73*73*64 output 71*71*80
            net = slim.conv2d(net, 80, [1, 1], scope="Conv2d_3b_1x1")
            # layer 6 input 71*71*80 output 71*71*192
            net = slim.conv2d(net, 192, [3 ,3], scope="Conv2d_4a_3x3")
            # layer 7 input 71*71*192 output 35*35*192
            net = slim.max_pool2d(net, [3, 3], stride=2, scope="MaxPool5a_3x3")
        
        # 第一个inception块
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # 第一个inception块 第一个部分
            with tf.variable_scope('Mixed_5b'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5],
                                           scope='Conv2d_0b_5x5')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope='Conv2d_0c_3x3')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                
            # 第二个部分
            with tf.variable_scope('Mixed_5c'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5],
                                           scope='Conv_1_0c_5x5')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope='Conv2d_0b_3x3')
                    
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope='Conv2d_0c_3x3')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1],
                                           scope='Conv2d_0b_1x1')
                
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                
            # 第三个部分
            with tf.variable_scope('Mixed_5d'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5],
                                           scope='Conv2d_0b_5x5')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], 
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3],
                                           scope='Conv2d_0c_3x3')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                
            # 第二个inception块 第一部分
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
                
            # 第二部分
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                
            # 第三部分
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                
            # 第四部分 与0c完全相同
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                
            # 第五部分 与0c完全相同
            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7],
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1],
                                           scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1],
                                           scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7],
                                           scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1],
                                           scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7],
                                           scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                
            end_points['Mixed_6e'] = net
            
            # 第三个Inception块 第一部分
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], 
                                           scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], 
                                           scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
                
            # 第二部分
            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = \
                    tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                               slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = \
                    tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                               slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                
            # 第三部分
            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = \
                    tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                               slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = \
                    tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                               slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1],
                                           scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            
            return net, end_points


def inception_v3(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
    
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes],
                           reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = inception_v3_base(inputs, scope=scope)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            aux_logits = end_points['Mixed_6e']
            with tf.variable_scope('AuxLogits'):
                aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID',
                                             scope='AvgPool_1a_5x5')
                aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                         scope='Conv2d_1b_1x1')
                
                aux_logits = slim.conv2d(aux_logits, 768, [5, 5], 
                                         weights_initializer=trunc_normal(0.01),
                                         padding='VALID', scope='Conv2d_2a_5x5')
                aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1],
                                         activation_fn=None, normalizer_fn=None,
                                         weights_initializer=trunc_normal(0.001),
                                         scope='Conv2d_2b_1x1')
                if spatial_squeeze:
                    aux_logits = tf.squeeze(aux_logits, [1, 2],
                                            name='SpatialSqueeze')
                end_points['AuxLogits']=aux_logits
                
            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, [8, 8], padding='VALID',
                                      scope='AvgPool_1a_8x8')
                net = slim.dropout(net, keep_prob=dropout_keep_prob,
                                   scope='Dropout_1b')
                end_points['PreLogits'] = net
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeeze')
                end_points['logits'] = logits
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
                
        return logits,end_points


# 定义评估每轮计算时间的函数
def time_tensorflow_run(session, target, info_string):
    
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_suqred = 0.0
    
    for i in range(num_batches+num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print ('%s:step %d,duration=%.3f' \
                       % (datetime.now(), i-num_steps_burn_in, duration))
                pass
            total_duration += duration
            total_duration_suqred += duration*duration
            
            mn = total_duration / num_batches  
            vr = total_duration_suqred / num_batches - mn*mn  
            sd = math.sqrt(vr)       
            print ('%s:%s across %d steps,%.3f +/- %.3f sec/batch ' \
                   % (datetime.now(), info_string, num_batches, mn, sd))
            pass
        pass
    pass

            
if __name__=='__main__':
    
    tf.reset_default_graph()
    
    batch_size = 32
    height, width = 299, 299
    inputs = tf.random_uniform((batch_size, height, width, 3))
    
    with slim.arg_scope(inception_v3_arg_scope()):
        logits, end_points = inception_v3(inputs, is_training=False)
        
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    num_batches = 100
    time_tensorflow_run(sess, logits, 'Forward')
    
    
