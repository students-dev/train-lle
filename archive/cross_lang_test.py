import train_lle

def test():
    model = train_lle.load_model('model_js.lle')
    input_data = [1, 2, 3, 4]
    pred = model.predict(input_data)
    print('Prediction:', pred.data)
    expected = [0.5, 0.6]
    if abs(pred.data[0] - expected[0]) < 0.01:
        print('Test passed')
    else:
        print('Test failed')

if __name__ == '__main__':
    test()