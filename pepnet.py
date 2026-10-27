import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding
from tensorflow.keras.models import Model
import numpy as np

class GateNU(Layer):
    """Gate Neural Unit - 基础门控单元"""
    
    def __init__(self, hidden_dim, output_dim, gamma=2.0, **kwargs):
        super(GateNU, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        
    def build(self, input_shape):
        # 第一层网络
        self.dense1 = Dense(self.hidden_dim, activation='relu', name='gate_nu_dense1')
        # 第二层网络
        self.dense2 = Dense(self.output_dim, activation='sigmoid', name='gate_nu_dense2')
        super(GateNU, self).build(input_shape)
        
    def call(self, inputs):
        # 第一层：特征交叉
        x_prime = self.dense1(inputs)
        # 第二层：生成门控分数
        delta = self.dense2(x_prime)
        # 缩放门控分数到 [0, gamma] 范围
        delta = self.gamma * delta
        return delta

class EPNet(Layer):
    """Embedding Personalized Network - 嵌入个性化网络"""
    
    def __init__(self, ep_hidden_dim, **kwargs):
        super(EPNet, self).__init__(**kwargs)
        self.ep_hidden_dim = ep_hidden_dim
        
    def build(self, input_shape):
        # 输入形状: [(batch_size, domain_features_dim), (batch_size, general_embedding_dim)]
        _, general_embedding_shape = input_shape
        
        # Gate NU 用于EPNet
        self.gate_nu = GateNU(hidden_dim=self.ep_hidden_dim, output_dim=general_embedding_shape[-1], name='epnet_gate_nu')
        super(EPNet, self).build(input_shape)
        
    def call(self, inputs):
        domain_features, embeddings = inputs
        
        # 拼接领域特征和嵌入（停止梯度）
        gate_input = tf.concat([
            domain_features, 
            tf.stop_gradient(embeddings)
        ], axis=-1)
        
        # 生成领域特定的门控
        domain_gate = self.gate_nu(gate_input)
        
        # 对原始嵌入进行个性化选择
        personalized_embeddings = domain_gate * embeddings
        
        return personalized_embeddings


class PPNet(Layer):
    """Parameter Personalized Network - 参数个性化网络"""
    
    def __init__(self, pp_hidden_dim, hidden_dim, **kwargs):
        super(PPNet, self).__init__(**kwargs)
        self.pp_hidden_dim = pp_hidden_dim
        self.hidden_dim = hidden_dim
        
    def build(self, input_shape):
        # 输入形状: [(batch_size, personal_embedding_dim), (batch_size, general_embedding_dim)]
        personal_embedding_shape, general_embedding_shape = input_shape
        
        # Gate NU 用于PPNet
        self.gate_nu = GateNU(
            hidden_dim=self.pp_hidden_dim,
            output_dim=self.hidden_dim, 
            name='ppnet_gate_nu'
        )
        super(PPNet, self).build(input_shape)
        
    def call(self, inputs):
        prior_features, embeddings = inputs
        
        # 拼接先验特征和嵌入（停止梯度）
        gate_input = tf.concat([
            prior_features,
            tf.stop_gradient(embeddings)
        ], axis=-1)
        
        # 生成任务特定的门控
        gate_score = self.gate_nu(gate_input)
        
        return gate_score


class PEPNetModel(Model):    
    def __init__(
                self, 
                hidden_units=[512, 128, 32],
                pp_hidden_dim=512,
                ep_hidden_dim=1024,
                num_tasks=2,
                num_domains=1,
                **kwargs
        ):
        super(PEPNetModel, self).__init__(**kwargs)
        
        self.hidden_units = hidden_units
        self.num_tasks = num_tasks
        self.num_domains = num_domains
        self.pp_hidden_dim = pp_hidden_dim
        self.ep_hidden_dim = ep_hidden_dim

        self.epnet = EPNet(ep_hidden_dim=ep_hidden_dim, name='epnet')
        
        # 任务塔
        self.task_towers = []
        self.ppnet_gate = []
        for i in range(num_tasks):
            tower_layers = []
            ppnet_layers = []
            for j, units in enumerate(hidden_units):
                layer_name = f'task{i}_dense{j}'
                tower_layers.append(Dense(units, activation='relu', name=layer_name))
                ppnet_layers.append(PPNet(pp_hidden_dim, units, name=f'task{i}_ppnet_gate{j}'))
            # 输出层
            tower_layers.append(Dense(1, activation=None, name=f'task{i}_output'))
            self.task_towers.append(tower_layers)
            self.ppnet_gate.append(ppnet_layers)
    
    def call(self, general_features, domain_features, personal_features, training=None):
        # 输入格式: {
        #   'general_features': sparse_features, dense_features
        #   'domain_features': tensor,  # 领域特定特征
        #   'personal_features': tensor,   # 个性化先验特征
        # }
        
        # EPNet: 嵌入个性化
        ep_output = self.epnet([domain_features, general_features])
        
        # 任务塔预测
        predictions = []

        for task_idx in range(self.num_tasks):
            # 每个任务有自己的DNN塔
            hidden = ep_output
            ppnet_layers = self.ppnet_gate[task_idx]
            for layer_idx, layer in enumerate(self.task_towers[task_idx][:-1]):
                hidden = layer(hidden)
    
                # 应用PPNet门控（除了最后一层）
                if layer_idx < len(ppnet_layers):
                    gate = ppnet_layers[layer_idx]([personal_features, ep_output])
                    hidden = hidden * gate

            # 最后一层输出
            output = self.task_towers[task_idx][-1](hidden)
            predictions.append(output)
        
        return predictions