# 第一次大作业——代码说明

### 几种算法的命名

- `square_parzen`——方核parzen
- `gauss_parzen`——高斯核parzen
- `sphere_parzen`——球核parzen
- `knn`——k临近估计

### 各个文件的作用

- `generate_sample.m`生成样本
- `main_xxx.m`——单独运行某个算法的主程序，并输出结果
- `xxx_predict.m`——核函数
- `run_exp_nsize.m`——实验脚本

> 由于时间关系，代码没有做重构，请见谅



### 运行

所有代码在matlab2016B环境上经过测试

在`run_exp_nsize.m`中修改样本数量，并在命令行运行该脚本即可：

```matlab
run_exp_nsize
```

> 如果想要单独运行某个算法，把其他算法的调用代码注释掉即可