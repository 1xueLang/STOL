```python main.py --dataset DVSGesture128 --T 20 -V --arch vgg11_bn -A --weight_decay 5e-3 --print_intv 10 --learning_rate 1e-2 --batch_size1 32 --batch_size2 64 --drop 0.5 --data_dir data_dir/DVS128Gesture/ --num_classes 11```
98.61 98.61. 98.23, 97.92 97.92


```python main.py -V --batch_size1 64 --learning_rate 0.05``` 95.47


```python main.py --arch vgg11_bn -V --batch_size1 64 --learning_rate 0.025 --weight_decay 0.0005 --dataset CIFAR10DVS --data_r ./data_dir/CIFAR10DVS/ --T 10 --drop 0.3 --tau 4 --num_epochs 300``` 84.3

```python main.py -V --dataset CIFAR100 --learning_rate 2e-2 --weight_decay 5e-4 --num_classes 100 --num_epochs 300``` 74.71