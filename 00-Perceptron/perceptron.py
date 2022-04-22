from functools import reduce


class Perceptron:
    def __init__(self, input_num, activator) -> None:
        """_summary_
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        Args:
            input_num (_type_): _description_
            activator (_type_): _description_
        """
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        #  偏置项初始化为0
        self.bias = 0.0
        self.input_num = input_num

    def __str__(self) -> str:
        """        
        打印学习到的权重、偏置项
        Returns:
            str: _description_
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """_summary_
        输入向量，输出感知器的计算结果
        Args:
            input_vec (_type_): _description_
        把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        变成[(x1,w1),(x2,w2),(x3,w3),...]

        """
        sum = 0
        for x, w in zip(input_vec, self.weights):
            sum += x*w
            sum += self.bias
        return self.activator(sum)

    def train(self, input_vecs, labels, iteration, learning_rate):
        """_summary_
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        Args:
            input_vecs (_type_): _description_
            labels (_type_): _description_
            iteration (_type_): _description_
            learning_rate (_type_): _description_
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, learning_rate)

    def _one_iteration(self, input_vecs, labels, learning_rate):
        """_summary_
        一次迭代，把所有的训练数据过一遍
        把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        而每个训练样本是(input_vec, label)
        Args:
            input_vecs (_type_): _description_
            labels (_type_): _description_
            learning_rate (_type_): _description_
        """

        for input_vec, label in zip(input_vecs, labels):
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, learning_rate)

    def _update_weights(self, input_vec, output, label, learning_rate):
        '''
        按照感知器规则更新权重
        把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        变成[(x1,w1),(x2,w2),(x3,w3),...]
        然后利用感知器规则更新权重
        '''

        delta = label - output
        self.weights = [w + learning_rate * delta *
                        x for x, w in zip(input_vec, self.weights)]

        # 更新bias
        self.bias += learning_rate * delta
