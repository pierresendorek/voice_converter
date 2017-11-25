import operator

class PiecewiseLinearFunction:

    def __init__(self, params=None):
        assert params is not None
        self.time_list = []
        self.value_list = []
        self.params = params

    def add_point(self, time=None, value=None):
        if self.time_list != []:
            assert time > self.time_list[-1]

        self.time_list.append(time)
        self.value_list.append(value)


    def get_value(self, time=None):
        assert time is not None

        # Assuming the list is sorted
        lesser_than_values = list(filter(lambda i_t: i_t[1] < time, enumerate(self.time_list)))
        greater_than_values = list(filter(lambda i_t: i_t[1] >= time, enumerate(self.time_list)))

        #print("lesser_than_values", lesser_than_values)
        #print("greater_than_values", greater_than_values)

        if lesser_than_values == []:
            i_time_prev = 0
            time_prev = self.time_list[0]
        else:
            # greatest value smallest than time
            i_time_prev, time_prev = lesser_than_values[-1]

        if greater_than_values == []:
            i_time_next = len(self.time_list) - 1
            time_next = self.time_list[-1]
        else:
            # smallest value greater than time
            i_time_next, time_next = greater_than_values[0]


        if time_next == time_prev:
            betweenness = 0
        else:
            betweenness = (time - time_prev)/(time_next - time_prev) # is in [0, 1[

        return (1-betweenness) * self.value_list[i_time_prev] + betweenness * self.value_list[i_time_next]




if __name__ == "__main__":
    from params.params import get_params
    import numpy as np
    params = get_params()

    u = np.ones(3)

    plf = PiecewiseLinearFunction(params=params)
    plf.add_point(time=3, value=3*u)
    plf.add_point(time=4, value=7*u)
    plf.add_point(time=5, value=1*u)
    plf.add_point(time=7, value=6*u)

    v = plf.get_value(time=3.5)
    print(v)




