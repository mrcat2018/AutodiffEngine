import autodiff as ad
import numpy as np


def test_identity():
    x2 = ad.Variable(name="x2")
    y = x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_add_by_const():
    x2 = ad.Variable(name = "x2")
    y = 5 + x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


def test_sub_by_const():
    x2 = ad.Variable(name='x2')
    y = 3 - x2
    grad_x2, = ad.gradients(y, [x2])
    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, 3 - x2_val)
    assert np.array_equal(grad_x2_val, -np.ones_like(x2_val))


def test_neg():
    x1 = ad.Variable(name='x1')
    x2 = ad.Variable(name='x2')

    y = -x2 + x1
    
    grad_x1, grad_x2 = ad.gradients(y, [x1, x2])
    executor = ad.Executor([y, grad_x1, grad_x2])
    x2_val = 2 * np.ones(3)
    x1_val = 3 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict = {x1: x1_val, x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, -x2_val + x1_val)
    assert np.array_equal(grad_x2_val, -np.ones_like(x2_val))
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))


def test_mul_by_const():
    x2 = ad.Variable(name = "x2")
    y = 5 * x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val * 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val) * 5)


def test_div_two_vars():
    x1 = ad.Variable(name = 'x1')
    x2 = ad.Variable(name = 'x2')
    
    y = x1 / x2

    grad_x1, grad_x2 = ad.gradients(y, [x1, x2])

    executor = ad.Executor([y, grad_x1, grad_x2])
    x1_val = 2 * np.ones(3)
    x2_val = 5 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val= executor.run(feed_dict = {x1: x1_val, x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x1_val / x2_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val) / x2_val)
    assert np.array_equal(grad_x2_val, -x1_val / (x2_val * x2_val))


def test_div_by_const():
    x2 = ad.Variable(name = "x2")
    y = 5 / x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, 5 / x2_val)
    print(grad_x2_val)
    print(-5 / (x2_val * x2_val))
    assert np.array_equal(grad_x2_val, -5 / (x2_val * x2_val))


def test_add_two_vars():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x2 + x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
  
    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + x3_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))
    assert np.array_equal(grad_x3_val, np.ones_like(x3_val))


def test_mul_two_vars():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x2 * x3
    
    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})
 
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val)
    assert np.array_equal(grad_x3_val, x2_val)


def test_add_mul_mix_1():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x1 + x2 * x3 * x1
    
    grad_x1, grad_x2, grad_x3 = ad.gradients(y, [x1, x2, x3])
   
    executor = ad.Executor([y, grad_x1, grad_x2, grad_x3])
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x1 : x1_val, x2: x2_val, x3 : x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val) + x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val * x1_val)
    assert np.array_equal(grad_x3_val, x2_val * x1_val)


def test_add_mul_mix_2():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    x4 = ad.Variable(name = "x4")
    y = x1 + x2 * x3 * x4
    
    grad_x1, grad_x2, grad_x3, grad_x4 = ad.gradients(y, [x1, x2, x3, x4])
   
    executor = ad.Executor([y, grad_x1, grad_x2, grad_x3, grad_x4])
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x4_val = 4 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val, grad_x4_val = executor.run(feed_dict = {x1 : x1_val, x2: x2_val, x3 : x3_val, x4 : x4_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val * x4_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))
    assert np.array_equal(grad_x2_val, x3_val * x4_val)
    assert np.array_equal(grad_x3_val, x2_val * x4_val)
    assert np.array_equal(grad_x4_val, x2_val * x3_val)


def test_add_mul_mix_3():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    z = x2 * x2 + x2 + x3 + 3
    y = z * z + x3
    
    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    z_val = x2_val * x2_val + x2_val + x3_val + 3
    expected_yval = z_val * z_val + x3_val
    expected_grad_x2_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) * (2 * x2_val + 1)
    expected_grad_x3_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) + 1
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)


def test_grad_of_grad():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x2 * x2 + x2 * x3
    
    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    grad_x2_x2, grad_x2_x3 = ad.gradients(grad_x2, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3, grad_x2_x2, grad_x2_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val, grad_x2_x2_val, grad_x2_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    expected_yval = x2_val * x2_val + x2_val * x3_val
    expected_grad_x2_val = 2 * x2_val + x3_val 
    expected_grad_x3_val = x2_val
    expected_grad_x2_x2_val = 2 * np.ones_like(x2_val)
    expected_grad_x2_x3_val = 1 * np.ones_like(x2_val)

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)
    assert np.array_equal(grad_x2_x2_val, expected_grad_x2_x2_val)
    assert np.array_equal(grad_x2_x3_val, expected_grad_x2_x3_val)


def test_matmul_two_vars():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = ad.matmul_op(x2, x3)

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    
    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = np.array([[1, 2], [3, 4], [5, 6]]) # 3x2
    x3_val = np.array([[7, 8, 9], [10, 11, 12]]) # 2x3

    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    expected_yval = np.matmul(x2_val, x3_val)
    expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(x3_val))
    expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)


def test_log_op():
    x1 = ad.Variable(name = "x1")
    y = ad.log(x1)

    grad_x1, = ad.gradients(y, [x1])

    executor = ad.Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val= executor.run(feed_dict = {x1 : x1_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, np.log(x1_val))
    assert np.array_equal(grad_x1_val, 1 / x1_val)


def test_log_two_vars():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    y = ad.log(x1 * x2)

    grad_x1, grad_x2 = ad.gradients(y, [x1, x2])

    executor = ad.Executor([y, grad_x1, grad_x2])
    x1_val = 2 * np.ones(3)
    x2_val = 4 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict = {x1 : x1_val, x2: x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, np.log(x1_val * x2_val))
    assert np.array_equal(grad_x1_val, x2_val / (x1_val * x2_val))
    assert np.array_equal(grad_x2_val, x1_val / (x1_val * x2_val))


def test_exp_op():
    x1 = ad.Variable(name = "x1")
    y = ad.exp(x1)

    grad_x1, = ad.gradients(y, [x1])

    executor = ad.Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val= executor.run(feed_dict = {x1 : x1_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, np.exp(x1_val))
    assert np.array_equal(grad_x1_val, np.exp(x1_val))


def test_exp_mix_op():
    x1 = ad.Variable(name="x1")
    x2 = ad.Variable(name="x2")
    y = ad.exp(ad.log(x1 * x2) + 1)

    grad_x1, grad_x2 = ad.gradients(y, [x1, x2])

    executor = ad.Executor([y, grad_x1, grad_x2])
    x1_val = 2 * np.ones(3)
    x2_val = 4 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict = {x1 : x1_val, x2: x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, np.exp(np.log(x1_val * x2_val) + 1))
    assert np.array_equal(grad_x1_val, y_val * x2_val / (x1_val * x2_val))
    assert np.array_equal(grad_x2_val, y_val * x1_val / (x1_val * x2_val))


def test_reduce_sum():
    x1 = ad.Variable(name = "x1")
    y = ad.reduce_sum(x1)

    grad_x1, = ad.gradients(y, [x1])

    executor = ad.Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val= executor.run(feed_dict = {x1 : x1_val})
    
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, np.sum(x1_val))
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))


def test_reduce_sum_mix():
    x1 = ad.Variable(name = "x1")
    y = ad.exp(ad.reduce_sum(x1))

    grad_x1, = ad.gradients(y, [x1])

    executor = ad.Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val= executor.run(feed_dict = {x1 : x1_val})
    expected_y_val = np.exp(np.sum(x1_val))
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_y_val)
    assert np.array_equal(grad_x1_val, expected_y_val * np.ones_like(x1_val))

    y2 = ad.log(ad.reduce_sum(x1))
    grad_x2, = ad.gradients(y2, [x1])
    executor2 = ad.Executor([y2, grad_x2])
    y2_val, grad_x2_val = executor2.run(feed_dict={x1: x1_val})
    expected_y2_val = np.log(np.sum(x1_val))
    assert isinstance(y2, ad.Node)
    assert np.array_equal(y2_val, expected_y2_val)
    assert np.array_equal(grad_x2_val, (1/np.sum(x1_val)) * np.ones_like(x1_val))


def test_mix_all():
    x1 = ad.Variable(name="x1")
    y = 1/(1+ad.exp(-ad.reduce_sum(x1)))

    grad_x1, = ad.gradients(y, [x1])

    executor = ad.Executor([y, grad_x1])
    x1_val = 2 * np.ones(3)
    y_val, grad_x1_val= executor.run(feed_dict = {x1 : x1_val})
    expected_y_val = 1/(1+np.exp(-np.sum(x1_val)))
    expected_y_grad = expected_y_val * (1 - expected_y_val) * np.ones_like(x1_val)

    print(expected_y_grad)
    print(grad_x1_val)
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_y_val)
    assert np.sum(np.abs(grad_x1_val - expected_y_grad)) < 1E-10


def test_logistic():
    x1 = ad.Variable(name="x1")
    w = ad.Variable(name='w')
    y = 1/(1+ad.exp(-ad.reduce_sum(w * x1)))

    grad_w, = ad.gradients(y, [w])

    executor = ad.Executor([y, grad_w])
    x1_val = 3 * np.ones(3)
    w_val = 3 * np.zeros(3)
    y_val, grad_w_val = executor.run(feed_dict={x1: x1_val, w: w_val})
    expected_y_val = 1/(1 + np.exp(-np.sum(w_val * x1_val)))
    expected_y_grad = expected_y_val * (1 - expected_y_val) * x1_val

    print(expected_y_grad)
    print(grad_w_val)
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_y_val)
    assert np.sum(np.abs(grad_w_val - expected_y_grad)) < 1E-7


def test_log_logistic():
    x1 = ad.Variable(name="x1")
    w = ad.Variable(name='w')
    y = ad.log(1/(1+ad.exp(-ad.reduce_sum(w * x1))))

    grad_w, = ad.gradients(y, [w])

    executor = ad.Executor([y, grad_w])
    x1_val = 3 * np.ones(3)
    w_val = 3 * np.zeros(3)
    y_val, grad_w_val = executor.run(feed_dict={x1: x1_val, w: w_val})
    logistic = 1/(1+np.exp(-np.sum(w_val * x1_val)))
    expected_y_val = np.log(logistic)
    expected_y_grad = (1 - logistic) * x1_val

    print(expected_y_grad)
    print(grad_w_val)
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_y_val)
    assert np.sum(np.abs(grad_w_val - expected_y_grad)) < 1E-7


def test_logistic_loss():
    x = ad.Variable(name='x')
    w = ad.Variable(name='w')
    y = ad.Variable(name='y')

    h = 1 / (1 + ad.exp(-ad.reduce_sum(w * x)))
    L = y * ad.log(h) + (1 - y) * ad.log(1 - h)
    w_grad, = ad.gradients(L, [w])
    executor = ad.Executor([L, w_grad])

    y_val = 0
    x_val = np.array([2, 3, 4])
    w_val = np.random.random(3)

    L_val, w_grad_val = executor.run(feed_dict={x: x_val, y: y_val, w: w_val})

    logistic = 1 / (1 + np.exp(-np.sum(w_val * x_val)))
    expected_L_val = y_val * np.log(logistic) + (1 - y_val) * np.log(1 - logistic)
    expected_w_grad = (y_val - logistic) * x_val

    print(L_val)
    print(expected_L_val)
    print(expected_w_grad)
    print(w_grad_val)

    assert expected_L_val == L_val
    assert np.sum(np.abs(expected_w_grad - w_grad_val)) < 1E-9
