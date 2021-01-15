# coding=utf-8
from torch import nn

class ACP_REG_CNN(nn.Module):
    def __init__(self, ppsize, feasize, conv1_w, conv1_s, conv1_f, pool1_w, pool1_s, conv2_w, conv2_s, conv2_f,
                 pool2_w, pool2_s, n_hid, hid_func):
        super(ACP_REG_CNN, self).__init__()
        print('Using ACP_REG_CNN model!')
        self.ppsize = ppsize
        self.feasize = feasize
        self.conv1_w = conv1_w
        self.conv1_s = conv1_s
        self.conv1_f = conv1_f
        self.pool1_w = pool1_w
        self.pool1_s = pool1_s
        self.conv2_w = conv2_w
        self.conv2_s = conv2_s
        self.conv2_f = conv2_f
        self.pool2_w = pool2_w
        self.pool2_s = pool2_s
        self.n_hid = n_hid
        self.hid_func = hid_func
        self.layer1_out = (self.ppsize + (self.conv1_w // 2 * 2) - self.conv1_w) // self.conv1_s + 1
        self.layer2_out = (self.layer1_out + (self.pool1_w // 2 * 2) - self.pool1_w) // self.pool1_s + 1
        self.layer3_out = (self.layer2_out + (self.conv2_w // 2 * 2) - self.conv2_w) // self.conv2_s + 1
        self.layer4_out = (self.layer3_out + (self.pool2_w // 2 * 2) - self.pool2_w) // self.pool2_s + 1
        def conv_layer1(windows, feature_dim, stride_size, filter_num):
            return nn.Sequential(
                nn.Conv2d(1, out_channels=filter_num, kernel_size=(windows, feature_dim), stride=stride_size,
                          padding=(windows // 2, 0)),
                nn.BatchNorm2d(filter_num),
                nn.LeakyReLU(inplace=False))
        def conv_layer2(windows, feature_dim, stride_size, filter_num):
            return nn.Sequential(
                nn.Conv2d(feature_dim, out_channels=filter_num, kernel_size=(windows, 1), stride=stride_size,padding=(windows // 2, 0)),
                nn.BatchNorm2d(filter_num),
                nn.LeakyReLU(inplace=False))
        def avg_pool_layer(windows, stride_size):
            return nn.Sequential(nn.AvgPool2d(kernel_size=(windows, 1), stride=stride_size, padding=(windows // 2, 0)))
        def max_pool_layer(windows):
            return nn.Sequential(nn.MaxPool2d(kernel_size=(windows, 1), stride=None))
        def fc_layer(input_dim, hidn_num, hid_func):
            act_func = nn.LeakyReLU(inplace=False)
            if hid_func == 'Sigmoid':
                act_func = nn.Sigmoid()
            elif hid_func == 'ReLU':
                act_func = nn.ReLU(inplace=False)
            return nn.Sequential(
                nn.Linear(input_dim, hidn_num, bias=True),
                nn.BatchNorm1d(hidn_num),
                act_func)
        def output_layer(input_dim, output_num):
            return nn.Sequential(
                nn.Linear(input_dim, output_num, bias=False))
        self.conv1 = conv_layer1(self.conv1_w, self.feasize, self.conv1_s, self.conv1_f)
        self.avg_pool1 = avg_pool_layer(self.pool1_w, self.pool1_s)
        self.conv2 = conv_layer2(self.conv2_w, self.conv1_f, self.conv2_s, self.conv2_f)
        self.avg_pool2 = avg_pool_layer(self.pool2_w, self.pool2_s)
        self.max_pool = max_pool_layer(self.layer4_out)
        self.fc_layer1 = fc_layer(self.conv2_f, self.n_hid, self.hid_func)
        self.fc_layer4 = fc_layer(1024, 512, self.hid_func)
        self.fc_layer5 = fc_layer(512, 256, self.hid_func)
        self.output_layer = output_layer(256, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.avg_pool1(x)
        x = self.conv2(x)
        x = self.avg_pool2(x)
        x = self.max_pool(x).squeeze(3)
        x = x.squeeze(2)
        x = self.fc_layer1(x)
        x = self.fc_layer4(x)
        x = self.fc_layer5(x)
        x = self.output_layer(x)
        return x.squeeze(1)

class ACP_REG_MT_CNN(nn.Module):
    def __init__(self, ppsize, feasize, conv1_w, conv1_s, conv1_f, pool1_w, pool1_s, conv2_w, conv2_s, conv2_f,
                 pool2_w, pool2_s, n_hid, hid_func, n_out):
        super(ACP_REG_MT_CNN, self).__init__()
        print('Using ACP_REG_MT_CNN model!')
        self.ppsize = ppsize
        self.feasize = feasize
        self.conv1_w = conv1_w
        self.conv1_s = conv1_s
        self.conv1_f = conv1_f
        self.pool1_w = pool1_w
        self.pool1_s = pool1_s
        self.conv2_w = conv2_w
        self.conv2_s = conv2_s
        self.conv2_f = conv2_f
        self.pool2_w = pool2_w
        self.pool2_s = pool2_s
        self.n_hid = n_hid
        self.hid_func = hid_func
        self.n_out = n_out
        self.layer1_out = (self.ppsize + (self.conv1_w // 2 * 2) - self.conv1_w) // self.conv1_s + 1
        self.layer2_out = (self.layer1_out + (self.pool1_w // 2 * 2) - self.pool1_w) // self.pool1_s + 1
        self.layer3_out = (self.layer2_out + (self.conv2_w // 2 * 2) - self.conv2_w) // self.conv2_s + 1
        self.layer4_out = (self.layer3_out + (self.pool2_w // 2 * 2) - self.pool2_w) // self.pool2_s + 1
        def conv_layer1(windows, feature_dim, stride_size, filter_num):
            return nn.Sequential(
                nn.Conv2d(1, out_channels=filter_num, kernel_size=(windows, feature_dim), stride=stride_size,
                          padding=(windows // 2, 0)),
                nn.BatchNorm2d(filter_num),
                nn.LeakyReLU(inplace=False))
        def conv_layer2(windows, feature_dim, stride_size, filter_num):
            return nn.Sequential(
                nn.Conv2d(feature_dim, out_channels=filter_num, kernel_size=(windows, 1), stride=stride_size,
                          padding=(windows // 2, 0)),
                nn.BatchNorm2d(filter_num),
                nn.LeakyReLU(inplace=False))
        def avg_pool_layer(windows, stride_size):
            return nn.Sequential(nn.AvgPool2d(kernel_size=(windows, 1), stride=stride_size, padding=(windows // 2, 0)))
        def max_pool_layer(windows):
            return nn.Sequential(nn.MaxPool2d(kernel_size=(windows, 1), stride=None))
        def fc_layer(input_dim, hidn_num, hid_func):
            act_func = nn.LeakyReLU(inplace=False)
            if hid_func == 'Sigmoid':
                act_func = nn.Sigmoid()
            elif hid_func == 'ReLU':
                act_func = nn.ReLU(inplace=False)
            return nn.Sequential(
                nn.Linear(input_dim, hidn_num, bias=True),
                nn.BatchNorm1d(hidn_num),
                act_func)
        def output_layer(input_dim, output_num):
            return nn.Sequential(nn.Linear(input_dim, output_num, bias=False))
        self.conv1 = conv_layer1(self.conv1_w, self.feasize, self.conv1_s, self.conv1_f)
        self.avg_pool1 = avg_pool_layer(self.pool1_w, self.pool1_s)
        self.conv2 = conv_layer2(self.conv2_w, self.conv1_f, self.conv2_s, self.conv2_f)
        self.avg_pool2 = avg_pool_layer(self.pool2_w, self.pool2_s)
        self.max_pool = max_pool_layer(self.layer4_out)
        self.fc_layer1 = fc_layer(self.conv2_f, self.n_hid, self.hid_func)
        self.fc_layer4 = fc_layer(1024, 512, self.hid_func)
        self.fc_layer5 = fc_layer(512, 256, self.hid_func)
        self.output_layer = output_layer(256, self.n_out)
    def forward(self, x):
        x = self.conv1(x)
        x = self.avg_pool1(x)
        x = self.conv2(x)
        x = self.avg_pool2(x)
        x = self.max_pool(x).squeeze(3)
        x = x.squeeze(2)
        x = self.fc_layer1(x)
        x = self.fc_layer4(x)
        x = self.fc_layer5(x)
        x = self.output_layer(x)
        return x.squeeze(1)