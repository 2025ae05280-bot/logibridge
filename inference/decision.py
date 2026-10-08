from collections import deque


class Debouncer:
    def __init__(self, escalation_count=2, escalation_window=3, deescalation_count=3):
        self.history = deque(maxlen=escalation_window)
        self.lower_count = 0
        self.current = 0
        self.escalation_count = escalation_count
        self.deescalation_count = deescalation_count

    def decide(self, probs, temp_mean, setpoint=4.0):
        model_class = int(max(range(len(probs)), key=probs.__getitem__))
        if abs(float(temp_mean) - setpoint) > 3.0:
            self.current = 2
            self.history.clear()
            self.lower_count = 0
            return model_class, 2, "interlock"
        self.history.append(model_class)
        if model_class > self.current and sum(c >= model_class for c in self.history) >= self.escalation_count:
            self.current = model_class
            self.lower_count = 0
            return model_class, self.current, "model"
        if model_class < self.current:
            self.lower_count += 1
            if self.lower_count >= self.deescalation_count:
                self.current = model_class
                self.lower_count = 0
        else:
            self.lower_count = 0
        return model_class, self.current, "model"
