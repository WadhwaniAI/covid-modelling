def get_recovery_loss(predicted_value, actual_value, method='mape'):
    if method == 'mape':
        return abs(predicted_value - actual_value)*100.0/actual_value