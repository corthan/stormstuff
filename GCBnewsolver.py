#!/usr/bin/env python3
#
# Solve the storm problem using a different data structure

import argparse
import logging


WEATHERHEADER = "xid,yid,date_id,hour,wind\n"
MESSAGE_COUNT = 100000
MIN_HOUR = 3
MAX_HOUR = 21
STEPS_PER_HOUR = 30
penalty = 720


class Score(object):

    def __init__(self):
        self.score = 0

    def get_score(self):
        return self.score

    def update_score(self, this_best):
        self.score = self.score + this_best


class Cell(object):

    def __init__(self, x, y, t, prob):
        self.started = False
        self.not_finished = True
        self.conf_in = prob
        self.conf_out = prob
        self.parent = [None, None]
        self.time = t
        self.x = x
        self.y = y
        self.this_wind = 0.0
        self.conf_out = self.conf_in * self.this_wind
        self.history = []
        self.history.append([self.conf_out, self.time, self.parent])

    def get_prob_out(self):
        return self.conf_out

    def reset_prob_out(self):
        self.conf_out = self.conf_in * self.this_wind

    def set_prob_in(self, p_in, t, last_x, last_y):
        self.conf_in = p_in
        self.conf_out = self.conf_in * self.this_wind
        self.time = t
        self.parent = last_x, last_y
        h = [self.conf_out, self.time, self.parent]
        self.history.append(h)

    def set_wind_confidence(self, layers, l):
        self.this_wind = layers[l][self.x][self.y]

    def get_parent(self):
        return self.parent

    def get_time(self):
        return self.time

    def get_history(self):
        return self.history


def layer_probs(wind_cells,calc_risk):
    """
    for every wind velocity, set probability of surviving via lookup table
    probs = [[0.0, 1.0],
             [7.0, 1.0],
             [7.5, 0.99],
             [8.0, 0.98],
             [8.5, 0.97],
             [9.0, 0.96],
             [9.5, 0.95],
             [10.0, 0.94],
             [10.5, 0.93],
             [11.0, 0.92],
             [11.5, 0.91],
             [12.0, 0.90],
             [12.5, 0.88],
             [13.0, 0.86],
             [13.5, 0.84],
             [14.0, 0.80],
             [14.5, 0.70],
             [15.0, 0.70],
             [17.0, 0.70],
             [25.0, 0.75],
             [1000, 0.9]]
    """
    probs = []
    this_conf = (0.0, 1.0)
    probs.append(this_conf)
    wind_value = 0.25
    for risk in calc_risk:
        conf = 1.0 - risk
        # cut off at wind value 25 BUT leave a small possibility of getting through
        if wind_value > 25.0:
            conf = 0.02
        this_conf = (wind_value, conf)
        probs.append(this_conf)
        wind_value += 0.5
    # add final value at wind value 1000 -> all values above 30.25 at 0
    this_conf = (1000.0, 0.0)
    probs.append(this_conf)
    # calc new celle values with confidence levels instead of wind values
    new_cells = wind_cells
    for r in range(len(wind_cells)):
        for x in range(len(wind_cells[r])):
            for y in range(len(wind_cells[r][x])):
                for p in range(len(probs)-1):
                    if wind_cells[r][x][y] >= probs[p][0] and  wind_cells[r][x][y] < probs[p+1][0] :
                        new_cells[r][x][y] = probs[p][1] + \
                        (abs(probs[p+1][1] - probs[p][1]) * abs(wind_cells[r][x][y] - probs[p][0]) / abs(probs[p+1][0] - probs[p][0]))
    return new_cells


def prob_solver(cells, cities):
    prob_values = []
    start_city = cities[0]
    steps_per_layer = STEPS_PER_HOUR
# setup initial board: 
    xsize = len(cells[0])
    ysize = len(cells[0][0])
    for x in range(xsize):
        prob_values.append([])
        for y in range(ysize):
            prob_values[x].append(Cell(x, y, 0, 0))
            prob_values[x][y].set_wind_confidence(cells, 0)
# start walk at first city so automatic input confidence at 1
    prob_values[start_city[0]][start_city[1]].set_prob_in(1.0, 0, start_city[0], start_city[1])
    count = 0
    level = 0
    total_steps = steps_per_layer * len(cells)
    for t in range(total_steps):
        switch_on(prob_values, t, cities, total_steps)
        count += 1
        if count == steps_per_layer:
            level += 1
            print("entering next hour", level)
            if level <= len(cells):
                prob_values = do_transfer(prob_values, t, cells, level)
                """
                if there is a value in delaying the launch time this would reset the confidence of the start city
                prob_values[start_city[0]][start_city[1]].set_prob_in(1.0, 0, start_city[0], start_city[1]) 
                """
            count = 0
        else:
            prob_values = take_step(prob_values, t, xsize, ysize)
    # now retrace paths
    all_paths = []
    for cit_no in range(len(cities)):
        if cit_no > 0:
            end_city = cities[cit_no]
            history = prob_values[end_city[0]][end_city[1]].get_history()
            path = find_path(prob_values, end_city)
            all_paths.append(path)
            print(history)
    return all_paths


def switch_on(prob_v, t, cities, total_time):
    start_city_x = cities[0][0]
    start_city_y = cities[0][1]
    for n in prob_v:
        for cell in n:
            x = cell.x
            y = cell.y
            steps_from_start = abs(x - start_city_x) + abs(y - start_city_y)
            if t == steps_from_start :
                cell.started = True
    # check for switch-off
            count = 0
            steps_from_closest_city = 2000
            for city in cities:
                if count > 0:
                    steps_from_this_city = abs(x - city[0]) + abs(y - city[1])
                    if steps_from_this_city < steps_from_closest_city:
                        steps_from_closest_city = steps_from_this_city
                count += 1
            if total_time - t == steps_from_closest_city:
                cell.not_finished = False
                x = cell.x
                y = cell.y
                p = cell.conf_in
                cell.set_prob_in(p, t, x, y)


def take_step(prob_v, t, xsize, ysize):
    """
    Take a step from x, y
    """
    xs = xsize
    ys = ysize
    tmpprobs = []
    for x in range(xs):
        tmpprobs.append([])
        for y in range(ys):
            tmpprobs[x].append([])
    for n in prob_v:
        for cell in n:
            x = cell.x
            y = cell.y
            p = cell.conf_in
            tmpprobs[x][y] = p, None, None
    for n in prob_v:
        for cell in n:
            if cell.started and cell.not_finished:
                x = cell.x
                y = cell.y
                step_prob = cell.conf_out
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if abs(dx) == abs(dy):  # no diags
                            continue
                        nx = x + dx
                        ny = y + dy
                        if nx < 0 or \
                                nx >= xs or \
                                ny < 0 or \
                                ny >= ys:
                            continue
                        # ok, nx, ny on board
                        if step_prob > tmpprobs[nx][ny][0]:
                            tmpprobs[nx][ny] = step_prob, x, y
    for n in prob_v:
        for cell in n:
            x = cell.x
            y = cell.y
            p = cell.conf_in
            if tmpprobs[x][y][0] > p:
                cell.set_prob_in(tmpprobs[x][y][0], t, tmpprobs[x][y][1], tmpprobs[x][y][2])
    return prob_v


def do_transfer(prob_v, t, area, level):
    """
    reset wind values
    reset conf_in for cells 'at home'
    reset conf_out for all cells
    """
    l = level
    for n in prob_v:
        for cell in n:
            prob_out = cell.get_prob_out()
            if cell.started and cell.not_finished:
                # cell.set_prob_in(prob_out, t, cell.x, cell.y) ///// uses too much memory
                # cell.conf_out = cell.conf_in
                cell.conf_in = prob_out
    take_step(prob_v, t, len(prob_v), len(prob_v[0]))
    for n in prob_v:
        for cell in n:
            if l < len(area):
                if cell.started and cell.not_finished:
                    cell.set_wind_confidence(area, l)
                    cell.reset_prob_out()
    return prob_v


def find_path(prob_v, end_city):
    history = prob_v[end_city[0]][end_city[1]].get_history()
# get best values and times INCLUDING PENALTY  - formula = t*conf + penalty * (1 - penalty)
    first = []
    first.append([0.0, 0, None, None])
    second = []
    second.append([0.0, 0, None, None])
    third = []
    third.append([0.0, 0, None, None])
    first_weighted_time = penalty
    second_weighted_time = penalty
    third_weighted_time = penalty
    for item in history:
        if item[0] * item[1] > 0.0000000001 or first_weighted_time != penalty:
            this_weighted_time = item[0] * item[1] + penalty * (1 - item[0])
            first_weighted_time = first[0][0] * first[0][1] + penalty * (1 - first[0][0])
            second_weighted_time = second[0][0] * second[0][1] + penalty * (1 - second[0][0])
            third_weighted_time = third[0][0] * third[0][1] + penalty * (1 - third[0][0])
            if this_weighted_time < first_weighted_time:
                third[0] = second[0]
                second[0] = first[0]
                first[0] = item
            else:
                if this_weighted_time < second_weighted_time:
                    third[0] = second[0]
                    second[0] = item
                else:
                    if this_weighted_time < third_weighted_time:
                        third[0] = item
        else:
            this_w_time = item[0] * item[1]
            first_w_time = first[0][0] * first[0][1]
            if this_w_time < first_w_time:
                first[0] = item
    print(first_weighted_time, second_weighted_time, third_weighted_time)
    # score.update_score(first_weighted_time)
    # print(first, second, third)
    this_back_track = first[0]
    t = first[0][1]
    while t > 0:
        hist_last_step = prob_v[this_back_track[2][0]][this_back_track[2][1]].get_history()
        for z in range(len(hist_last_step) - 1):
            this_ht = hist_last_step[z][1]
            next_ht = hist_last_step[z+1][1]
            if t > this_ht and t <= next_ht:
                next_back_track = hist_last_step[z]
                first.append(next_back_track)
                this_back_track = next_back_track
                t = this_ht

    return first


def out_put_file(best_path, end_city):
    length = len(best_path)
    outputfile = []
    this_step_time = 0
    last_step_time = 0
    # put in start point
    output_list = [3, 0, best_path[-1][2]]
    outputfile.append(output_list)
    for inst in range(length):
        now = length - 1 - inst
        this_step_time = best_path[now][1]
        this_step_xy = best_path[now][2]
        for n in range(this_step_time - last_step_time):
            actual_step = last_step_time + n + 1
            actual_minutes = actual_step * 2  # two minutes per step
            hours_from_start = int(actual_minutes / 60)
            output_minutes = actual_minutes - (hours_from_start * 60)
            output_hours = 3 + hours_from_start
            output_list = [output_hours, output_minutes, this_step_xy]
            outputfile.append(output_list)
        last_step_time = this_step_time
    # now add city position
    actual_step = this_step_time + 1
    actual_minutes = actual_step * 2  # two minutes per step
    hours_from_start = int(actual_minutes / 60)
    output_minutes = actual_minutes - (hours_from_start * 60)
    output_hours = 3 + hours_from_start
    output_list = [output_hours, output_minutes, end_city]
    outputfile.append(output_list)

    return outputfile


class Weighter(object):
    """
    This reads probabilities from a file, then
    uses this to map winds to probs
    The data is a single line of probabilities for buckets of 0.5 each
    """

    Nprobs = 60
    BucketWidth = 0.5
    MAX = 1.0

    def __init__(self, args):
        self.values = None
        logging.warning("Reading probabilities ...")
        with open(args.probfile, "r") as f:
            line = f.readline()
            # skip comments as weight file has a lot of extra stuff
            while line != '' and line[0] == '#':
                line = f.readline()
            self.values = list(map(float, line[:-1].split(',')))
        assert len(list(self.values)) == Weighter.Nprobs, "Unexpected number of probabilities!"
        for v in self.values:
            assert v <= Weighter.MAX, "Bad value"

    def get_prob(self, wind):
        index = int(wind / Weighter.BucketWidth)
        if index >= Weighter.Nprobs:
            index = Weighter.Nprobs - 1
        return self.values[index]


def scan_file_for_dimensions(fn):
    mx = 0
    my = 0
    maxh = 0
    minh = 24  # hour cannot be beyond end of day
    count = 0
    logging.warning("Scanning input file for size information ...")
    with open(fn, "r") as f:
        line = f.readline()
        assert line == WEATHERHEADER, "Malformed file - header does not match expectations"
        line = f.readline()
        while line != '':
            if count % MESSAGE_COUNT == 0:
                print(count)
            count += 1
            # note dropping of final \n
            xid_r, yid_r, date_r, hour_r, wind_r = line[:-1].split(',')
            xid, yid, hid = int(xid_r), int(yid_r), int(hour_r)
            if xid > mx:
                mx = xid
            if yid > my:
                my = yid
            if hid > maxh:
                maxh = hid
            if hid < minh:
                minh = hid
            line = f.readline()
        return mx, my, minh, maxh


def read_layers(args):
    xsize, ysize, minh, maxh = scan_file_for_dimensions(args.weatherfile)
    assert minh >= MIN_HOUR, "Unexpected early hour!"
    assert maxh < MAX_HOUR, "Unexpected late hour!"
    # hsize = MAX_HOUR - MIN_HOUR
    logging.warning("Creating layer structure ...")
    layers = [[[0 for y in range(1, ysize + 1)] for x in range(1, xsize + 1)] for h in range(minh, maxh + 1)]
    logging.warning("Reading weather file data into layers...")
    with open(args.weatherfile, "r") as f:
        line = f.readline()
        assert line == WEATHERHEADER, "Unexpected header for weather data"
        line = f.readline()
        read_count = 0
        while line != '':
            if read_count % MESSAGE_COUNT == 0:
                print(read_count)
            read_count += 1
            xid_r, yid_r, date_id_r, hour_r, wind_r = line[:-1].split(',')
            xid, yid, hour, wind = int(xid_r) - 1, int(yid_r) - 1, int(hour_r) - MIN_HOUR, float(wind_r)
            layers[hour][xid][yid] = wind
            line = f.readline()
    return layers, xsize, ysize


def read_cities(args):
    tmp = {}
    with open(args.cities, "r") as f:
        line = f.readline()
        assert line == "cid,xid,yid\n", "Malformed city file"
        line = f.readline()
        while line != '':
            cid, xid, yid = list(map(int, line[:-1].split(',')))
            tmp[cid] = (xid - 1, yid - 1)
            line = f.readline()
    cities = []
    for k in sorted(tmp.keys()):
        cities.append(tmp[k])
    assert tmp[0] == cities[0], "Something wrong reading cities"
    return cities


def confidence(before, prob):
    return before * (1 - prob)


def get_furthest_city(cities):
    maxd = 0
    fc = None
    for c in cities[1:]:
        dist = abs(c[0] - cities[0][0]) + abs(c[1] - cities[0][1])
        if dist > maxd:
            maxd = dist
            fc = c
    return fc


def show_layers(lys):
    logging.info("Layers :")
    for l in lys:
        for r in l:
            logging.info(r)
        logging.info("-------------------------------")
    logging.info("================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weatherfile", default="combined_day_1.csv",
                        help="Weather prediction data in csv format")
    parser.add_argument("-d", "--day_number", default=1, help="day number")
    parser.add_argument("-c", "--cities", default="CityData.csv", help="City data in csv format")
    parser.add_argument("-o", "--output", default="output_path.csv", help="output info in csv format")
    parser.add_argument("-p", "--probfile", default="ProbData.csv", help="Mapping of wind to prob >= 15")
    parser.add_argument("-l", "--log", default='WARNING', help="Logging level to use.")
    args = parser.parse_args()

    nl = getattr(logging, args.log.upper(), None)
    if not isinstance(nl, int):
        raise ValueError("Invalid log level: {}".format(args.log))
    logging.basicConfig(level=nl,
                        format='%(message)s')
    layers, xsize, ysize = read_layers(args)
    cities = read_cities(args)
    weighter = Weighter(args)
    # start_time = MIN_HOUR * STEPS_PER_HOUR
    # furthest_city = get_furthest_city(cities)
    # score = Score
    show_layers(layers)
    # reset layers to probabilities
    prob_board = layer_probs(layers, weighter.values)
    prob_paths = prob_solver(prob_board, cities)
    print(prob_paths)
    # get output data
    city_number = 1
    total_out = []
    with open(args.output, "w") as f:
        for path in prob_paths:
            print("treating city", city_number)
            out_put = out_put_file(path, cities[city_number])
            print(out_put)
            for step in out_put:
                day = args.day_number
                hour = step[0]
                minutes = step[1]
                if step[2] != None:
                    x = step[2][0] + 1
                    y = step[2][1] + 1
                    line = "{},{},{}:{},{},{}\n".format(city_number, day, hour, minutes, x, y)
                    f.write(line)
            city_number += 1
            total_out.append(out_put)
    # print(total_out)
    # print(day)


if __name__ == '__main__':
    main()
