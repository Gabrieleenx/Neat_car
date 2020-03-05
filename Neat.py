from collections import namedtuple
import random
import math
import copy
import pickle


class Network(object):

    def __init__(self, trans, node_order, node_layer, node_inv_number, input_nodes, output_nodes, fitness):
        self.trans = trans
        self.node_order = node_order
        self.node_layer = node_layer
        self.node_inv_number = node_inv_number
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.fitness = fitness
        self.node_values = dict()

    def reset_node_values(self):
        for i in range(len(self.node_order)):
            self.node_values[self.node_order[i]] = 0


Transition = namedtuple(typename='Transition', field_names=['source', 'target', 'weight', 'inv_number', 'enabled'])


def filter_trans_by_source(trans, states_to_keep):
    return {t for t in trans if t.source in states_to_keep}


def filter_trans_by_target(trans, states_to_keep):
    return {t for t in trans if t.target in states_to_keep}


def filter_trans_by_enabled(trans, states_to_keep):
    return {t for t in trans if t.enabled in states_to_keep}


def filter_trans_by_inv_number(trans, states_to_keep):
    return {t for t in trans if t.inv_number == states_to_keep}


def extract_elems_from_trans(trans, field):
    return {getattr(t, field) for t in trans}


def calc_node_order(trans, node_layer, node_inv_number, all_nodes, max_layer_size):
    done = False
    current = all_nodes
    current_ = current
    done_nodes = set()
    trans_order = list()
    node_order = list()
    enabled_trans = filter_trans_by_enabled(trans, '1')

    while not done:
        for node in current:
            trans_ = filter_trans_by_target(enabled_trans, node)
            source_ = extract_elems_from_trans(trans_, 'source')

            if not source_.issubset(done_nodes):
                continue

            trans_ = filter_trans_by_source(enabled_trans, node)
            for transition in trans_:
                trans_order.append(transition.inv_number)
            done_nodes = done_nodes.union({node})
            current_ = current_.difference({node})
            node_order.append(node)

        if not current:
            done = True
            continue
        if current == current_:
            cur_min_lay = max_layer_size+1
            for node_ in current:
                if node_layer[node_] <= cur_min_lay:
                    if node_layer[node_] == cur_min_lay:
                        if node_inv_number[node_] < min_inv:
                            cur_min_lay = node_layer[node_]
                            skip_node = node_
                            min_inv = node_inv_number[node_]
                    else:
                        cur_min_lay = node_layer[node_]
                        skip_node = node_
                        min_inv = node_inv_number[node_]
            done_nodes = done_nodes.union({skip_node})
            current_ = current_.difference({skip_node})
            node_order.append(skip_node)
            trans_ = filter_trans_by_source(enabled_trans, skip_node)
            for transition in trans_:
                trans_order.append(transition.inv_number)

        current = current_

        if not current:
            done = True
    return node_order


def mutate_weights(trans, trans_mutations, weight_mutations, diff):
    transition_ = set()
    for transition in trans:
        if random.random() > trans_mutations:
            transition_ = transition_.union({transition})
            continue
        if random.random() > 0.9:
            weight = 0 + 1*diff*(2*random.random()-1)
        else:
            weight = transition.weight + weight_mutations*diff*(2*random.random()-1)
        if weight > diff:
            weight = diff
        elif weight < -diff:
            weight = -diff
        transition_ = transition_.union({Transition(transition.source, transition.target, weight, transition.inv_number,
                                                    transition.enabled)})

    return transition_


def calc_delta_species(individual, species_, c1, c2, c3):
    not_matching_genome = list()
    max_gen_1 = 0
    max_gen_2 = 0
    disjoint = 0
    excess = 0
    weight_diff = 0
    num_weight = 0
    for gen_1 in individual.trans:
        if gen_1.inv_number > max_gen_1:
            max_gen_1 = gen_1.inv_number
        match = filter_trans_by_inv_number(species_.trans, gen_1.inv_number)
        if not match:
            not_matching_genome.append(gen_1)
        else:
            weight_diff += abs(list(match)[0].weight - gen_1.weight)
            num_weight += 1
    for gen_2 in species_.trans:
        if gen_2.inv_number > max_gen_2:
            max_gen_2 = gen_2.inv_number
        if not filter_trans_by_inv_number(individual.trans, gen_2.inv_number):
            not_matching_genome.append(gen_2)
    min_max_gen = min([max_gen_1, max_gen_2])
    for gen in not_matching_genome:
        if gen.inv_number <= min_max_gen:
            disjoint += 1
        else:
            excess += 1
    n = max([len(individual.trans), len(species_.trans)])
    delta = c1 * excess / n + c2 * disjoint / n + c3 * weight_diff / num_weight
    return delta


def speciation( old_species, population, delta_species, c1, c2, c3):
    species = list()
    for i in range(len(old_species)):
        species.append([])
    for ind_num in population:
        individual = population[ind_num]
        found_species = False
        for index in range(len(old_species)):
            species_ = old_species[index]
            if individual == species_:
                species[index].append(individual)
                found_species = True
                break
            delta = calc_delta_species(individual, species_, c1, c2, c3)

            if delta <= delta_species:
                species[index].append(individual)
                found_species = True
                break
        if not found_species:
            species.append([individual])
            old_species.append(individual)
    return species, old_species


def check_extinction(species, species_fitness, species_gen_stalled, gen_to_extinct):
    while len(species_fitness) < len(species):
        species_fitness.append(0)
        species_gen_stalled.append(0)
    extinction = list()
    for specie_ in range(len(species)):
        update_ = False
        for individual_ in species[specie_]:
            if individual_.fitness > species_fitness[specie_]:
                species_fitness[specie_] = individual_.fitness
                species_gen_stalled[specie_] = 0
                update_ = True
        if not update_:
            species_gen_stalled[specie_] += 1

        if species_gen_stalled[specie_] >= gen_to_extinct:
            extinction.append(specie_)
    for del_ in sorted(extinction, reverse=True):
        del species[del_]
        del species_gen_stalled[del_]
        del species_fitness[del_]
    return species, species_fitness, species_gen_stalled


def population_adjusted_fitness_sum(specie, c1, c2, c3, delta_species):
    adjusted_sum_fitness = 0
    for specie_i in specie:
        sh = 0
        for specie_j in specie:

            if calc_delta_species(specie_i, specie_j, c1, c2, c3) < delta_species:
                sh += 1
        adjusted_sum_fitness += specie_i.fitness/sh
    return adjusted_sum_fitness


def cross_over(parent_1, parent_2, max_layer_size):
    input_nodes = parent_1.input_nodes
    output_nodes = parent_1.output_nodes
    check_inv = set()
    trans_new = set()
    all_nodes = set()
    node_inv_num = dict()
    node_layer = dict()
    if parent_1.fitness == parent_2.fitness:
        for transition_ in parent_1.trans:
            check_inv = check_inv.union({transition_.inv_number})
        for transition_ in parent_2.trans:
            check_inv = check_inv.union({transition_.inv_number})
    else:

        for transition_ in parent_1.trans:

            check_inv = check_inv.union({transition_.inv_number})
    for inv in check_inv:
        trans_1 = filter_trans_by_inv_number(parent_1.trans, inv)
        trans_2 = filter_trans_by_inv_number(parent_2.trans, inv)
        if trans_1 and trans_2:
            all_nodes = all_nodes.union({str(list(trans_1)[0].source), str(list(trans_2)[0].source),
                                         str(list(trans_1)[0].target), str(list(trans_2)[0].target)})
        elif trans_1:
            all_nodes = all_nodes.union({str(list(trans_1)[0].source),
                                         str(list(trans_1)[0].target)})
        else:
            all_nodes = all_nodes.union({str(list(trans_2)[0].source),
                                         str(list(trans_2)[0].target)})
        if not trans_1:
            node_layer.update({str(list(trans_2)[0].source):parent_2.node_layer[str(list(trans_2)[0].source)]})
            node_layer.update({str(list(trans_2)[0].target):parent_2.node_layer[str(list(trans_2)[0].target)]})
            trans_new = trans_new.union({list(trans_2)[0]})
        elif not trans_2:
            node_layer.update({str(list(trans_1)[0].source):parent_1.node_layer[str(list(trans_1)[0].source)]})
            node_layer.update({str(list(trans_1)[0].target):parent_1.node_layer[str(list(trans_1)[0].target)]})
            trans_new = trans_new.union({list(trans_1)[0]})
        else:
            node_layer.update({str(list(trans_1)[0].source): parent_1.node_layer[str(list(trans_1)[0].source)]})
            node_layer.update({str(list(trans_1)[0].target): parent_1.node_layer[str(list(trans_1)[0].target)]})
            node_layer.update({str(list(trans_2)[0].source): parent_2.node_layer[str(list(trans_2)[0].source)]})
            node_layer.update({str(list(trans_2)[0].target): parent_2.node_layer[str(list(trans_2)[0].target)]})
            trans_ = list(trans_1)[0]
            # check if eigther or both trans are disabled.
            if list(trans_1)[0].enabled == '0' and list(trans_2)[0].enabled == '0':
                trans__ = Transition(trans_.source, trans_.target,
                                     random.choice([list(trans_1)[0].weight, list(trans_2)[0].weight]),
                                     trans_.inv_number,
                                     '0')
            elif list(trans_1)[0].enabled == '0' or list(trans_2)[0].enabled == '0':
                if random.random() > 0.25:
                    trans__ = Transition(trans_.source, trans_.target,
                                         random.choice([list(trans_1)[0].weight, list(trans_2)[0].weight]),
                                         trans_.inv_number,
                                         '0')
                else:
                    trans__ = Transition(trans_.source, trans_.target,
                                         random.choice([list(trans_1)[0].weight, list(trans_2)[0].weight]),
                                         trans_.inv_number,
                                         '1')
            else:
                trans__ = Transition(trans_.source, trans_.target,
                                     random.choice([list(trans_1)[0].weight, list(trans_2)[0].weight]),
                                     trans_.inv_number,
                                     '1')
            trans_new = trans_new.union({trans__})
    for node in all_nodes:
        node_inv_num[node] = int(node)

    node_order = calc_node_order(trans_new, node_layer, node_inv_num, all_nodes, max_layer_size)
    child = Network(trans_new, node_order, node_layer, node_inv_num, input_nodes, output_nodes, 0)
    return child


def find_parents(species, p_survivors):
    fitness_in_specie = list()
    for specie_ in species:
        fitness_in_specie.append(specie_.fitness)
    orderd_index = sorted(range(len(fitness_in_specie)), key=lambda k: fitness_in_specie[k], reverse=True)
    specie_size = len(orderd_index)
    parents = list()
    if specie_size <= 3:
        parents.append(species[orderd_index[0]])
    else:
        num_survivors = round(specie_size * p_survivors)
        if num_survivors < 2:
            num_survivors = 2
        for i in range(num_survivors):
            parents.append(species[orderd_index[i]])
    return parents, orderd_index


def offspring_size(species_fitness, pop_size):
    offspring = list()
    for index_ in range(len(species_fitness)):
        offspring.append(round(pop_size * species_fitness[index_] / sum(species_fitness)))
    while sum(offspring) < pop_size:
        offspring[0] += 1
    return offspring


def add_node_mutation(child, trans_inv_num, node_inv_num, gen_mutations, max_layer_size):
    trans_ = random.choice(list(child.trans))
    if trans_.source+','+trans_.target in gen_mutations.keys():
        inv = gen_mutations[trans_.source+','+trans_.target]
        t1 = inv[0]
        t2 = inv[1]
        n = inv[2]
    else:
        t1 = trans_inv_num
        t2 = trans_inv_num+1
        n = node_inv_num
        gen_mutations.update({ trans_.source+','+trans_.target: [t1, t2, n]})
        node_inv_num += 1
        trans_inv_num += 2

    trans_old = Transition(trans_.source, trans_.target, trans_.weight, trans_.inv_number, '0')
    trans_new_1 = Transition(trans_.source, str(n), trans_.weight, t1, '1')
    trans_new_2 = Transition(str(n), trans_.target, 1, t2, '1')
    child.trans = child.trans.union({trans_old, trans_new_1, trans_new_2})
    child.trans = child.trans.difference({trans_})

    child.node_layer.update({str(n): child.node_layer[trans_.source] + int((child.node_layer[trans_.target]
                                                                        - child.node_layer[trans_.source])/2)})
    child.node_inv_number[str(n)] = n
    all_nodes = set()
    for j in child.node_layer:
        all_nodes = all_nodes.union({j})

    child.node_order = calc_node_order(child.trans, child.node_layer, child.node_inv_number, all_nodes, max_layer_size)

    return child, trans_inv_num, node_inv_num, gen_mutations


def add_link_mutation(child, trans_inv_num, node_inv_num, gen_mutations, max_layer_size):
    done = False
    while not done:
        source = random.choice(child.node_order)
        target = random.choice(child.node_order)
        if source != target:
            if not filter_trans_by_source(child.trans, source).intersection(filter_trans_by_source(child.trans, target)):
                done = True

    if source+','+target in gen_mutations.keys():
        inv = gen_mutations[source+','+target]
        t1 = inv[0]
        t2 = inv[1]
        n = inv[2]
    else:
        t1 = trans_inv_num
        t2 = trans_inv_num+1
        n = node_inv_num
        gen_mutations.update({source+','+target: [t1, t2, n]})
        trans_inv_num += 1

    trans_new = Transition(source, target, 4*random.random()-2, t1, '1')

    child.trans = child.trans.union({trans_new})

    all_nodes = set()
    for j in child.node_layer:
        all_nodes = all_nodes.union({j})

    child.node_order = calc_node_order(child.trans, child.node_layer, child.node_inv_number, all_nodes, max_layer_size)

    return child, trans_inv_num, node_inv_num, gen_mutations


def enable_disable(child, ratio, max_layer_size):

    if random.random() < ratio:
        trans_m = filter_trans_by_enabled(child.trans, '1')
        flip = '0'
    else:
        trans_m = filter_trans_by_enabled(child.trans, '0')
        flip = '1'
    if trans_m:
        trans_ = random.choice(list(trans_m))
        trans_new = Transition(trans_.source, trans_.target, trans_.weight, trans_.inv_number, flip)

    else:
        return child

    child.trans = child.trans.union({trans_new})
    child.trans = child.trans.difference({trans_})
    all_nodes = set()
    for j in child.node_layer:
        all_nodes = all_nodes.union({j})

    child.node_order = calc_node_order(child.trans, child.node_layer, child.node_inv_number, all_nodes, max_layer_size)
    return child


class Neat(object):

    def __init__(self, population_size, generations_to_extinct, c1, c2, c3, delta_species, input_size, output_size):
        self.pop_size = population_size
        self.gen_to_extinct = generations_to_extinct
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.delta_species = delta_species
        self.input_size = input_size
        self.out_size = output_size
        self.population = dict()
        self.trans_inv_num = 0
        self.node_inv_num = 0
        self.max_layer_size = 2**16-1
        self.species = list()
        self.old_species = list()
        self.species_fitness = list()
        self.species_gen_stalled = list()

    def initial_population(self):
        in_trans_set = set()
        all_nodes = set()
        input_nodes = set()
        output_nodes = set()
        node_inv_num = dict()
        node_layer = dict()
        for in_trans in range(self.input_size):
            node_layer[str(in_trans)] = 0
            input_nodes = input_nodes.union({str(in_trans)})
            for out_trans_ in range(self.out_size):
                out_trans = out_trans_ + self.input_size
                in_trans_set = in_trans_set.union({Transition(str(in_trans), str(out_trans), 0, self.trans_inv_num, '1')})
                self.trans_inv_num += 1
                all_nodes = all_nodes.union({str(in_trans), str(out_trans)})
                node_inv_num[str(in_trans)] = in_trans
                node_inv_num[str(out_trans)] = out_trans
                node_layer[str(out_trans)] = self.max_layer_size
                output_nodes = output_nodes.union({str(out_trans)})
        self.node_inv_num = len(node_inv_num)

        node_order = calc_node_order(in_trans_set, node_layer, node_inv_num, all_nodes,
                                                       self.max_layer_size)
        for pop in range(self.pop_size):
            trans = mutate_weights(in_trans_set, 1, 1, 2)
            self.population[str(pop)] = Network(trans=trans, node_order=node_order,
                                                node_layer=node_layer, node_inv_number=node_inv_num,
                                                input_nodes=input_nodes, output_nodes=output_nodes, fitness=0)

    def evaluate(self, network_nr, net_input, recursion):
        net = self.population[str(network_nr)]
        net.reset_node_values()
        output = [0] * len(net.output_nodes)
        for i in range(recursion):
            for j in range(len(net.node_order)):
                trans_ = filter_trans_by_target(trans=net.trans, states_to_keep=net.node_order[j])
                node_value = 0
                disabled = False
                for transition_ in trans_:
                    if transition_.enabled == '0':
                        disabled = True
                        continue
                    source_ = transition_.source
                    source_value = net.node_values[source_]
                    node_value += transition_.weight * source_value
                if {net.node_order[j]}.issubset(net.input_nodes):
                    net.node_values[net.node_order[j]] = net_input[int(net.node_order[j])] + \
                                                         float(1/(1+math.exp(-node_value))) - 0.5
                elif not (len(trans_) == 1 and disabled):
                    net.node_values[net.node_order[j]] = float(1/(1+math.exp(-node_value)))
        for output_ in net.output_nodes:
            output[int(output_)-len(net.input_nodes)] = net.node_values[output_]
        return output
    
    def save_net(self, network_nr):
        file_save = open('net_obj.obj', 'wb')
        pickle.dump(self.population[str(network_nr)], file_save)
        file_save.close()
        
    def load_net(self):  
        file_load = open('net_obj.obj', 'r')
        self.population['0'] = pickle.load(file_load)
        file_load.close()
    
    def update_fitness(self, network_nr, fitness_score):
        self.population[str(network_nr)].fitness = fitness_score
        # self.fitness_score[str(network_nr)] = fitness_score

    def train(self, p_survivors, weight_mutations, enable_ratio):
        new_population = dict()
        gen_mutations = dict()
        new_index = 0
        new_species = list()
        print('---before specie---')
        self.species, self.old_species = speciation(self.old_species, self.population, self.delta_species, self.c1, self.c2, self.c3)
        print('---after specie---')
        self.species, self.species_fitness, self.species_gen_stalled = \
            check_extinction(species=self.species, species_fitness=self.species_fitness,
                             species_gen_stalled=self.species_gen_stalled, gen_to_extinct=self.gen_to_extinct)
        print('---after extinct---')
        for index_ in range(len(self.species)):
            self.species_fitness[index_] = population_adjusted_fitness_sum(self.species[index_], self.c1, self.c2,
                                                                           self.c3, self.delta_species)
        print('---after pop adjusted---')
        offspring = offspring_size(self.species_fitness, self.pop_size)
        print('offspring', offspring)
        print('---after offspring size--+')
        extinction = list()
        for specie_index in range(len(offspring)):

            offspring_cnt = offspring[specie_index]

            if offspring_cnt == 0:
                extinction.append(specie_index)
                continue
            
            parents, orderd_index = find_parents(species=self.species[specie_index], p_survivors=p_survivors)
            # new_species.append(parents[0])
            i = 0
            j = 1
            k = 0
            parent_index = range(len(parents))
            new = True
            # print('offsprng_cnt', offspring_cnt)
            #for ii in range(len(orderd_index)):
            #    print('specis, index, fitness',specie_index,ii, self.species[specie_index][orderd_index[ii]].fitness)
            
            while offspring_cnt > 0:
                # print('new_index', new_index)
                offspring_cnt -= 1
                if offspring[specie_index] >= 5 and new:
                    new_population[str(new_index)] = self.species[specie_index][orderd_index[0]]
                    new_index += 1
                    new = False
                    continue

                if len(parents) >= 2 and random.random() > 0.025:
                    child = cross_over(parents[i], parents[j], self.max_layer_size)
                    i = random.choice(parent_index[:int(len(parents)/1.5)])
                    j = random.choice(parent_index[i+1:])
                else:
                    child = copy.deepcopy(parents[k])
                    k += 1
                    if k >= len(parents):
                        k = 0

                if random.random() < 0.03:
                    child, self.trans_inv_num, self.node_inv_num, gen_mutations = \
                        add_node_mutation(child, self.trans_inv_num, self.node_inv_num, gen_mutations, self.max_layer_size)
                if random.random() < 0.05:
                    child, self.trans_inv_num, self.node_inv_num, gen_mutations = \
                        add_link_mutation(child, self.trans_inv_num, self.node_inv_num, gen_mutations, self.max_layer_size)
                if random.random() < 0.01:
                    child = enable_disable(child, enable_ratio, self.max_layer_size)

                child.trans = mutate_weights(trans=child.trans, trans_mutations=0.8, weight_mutations=weight_mutations,
                                             diff=2)

                new_population[str(new_index)] = child
                new_index += 1

        for del_ in sorted(extinction, reverse=True):
            del self.species[del_]
            del self.old_species[del_]
            del self.species_gen_stalled[del_]
            del self.species_fitness[del_]

        self.population = new_population
        #self.old_species = new_species

'''

neat = Neat(population_size=50, generations_to_extinct=8, c1=1, c2=1, c3=0.4, delta_species=0.4, input_size=3,
            output_size=1)
neat.initial_population()

out = neat.evaluate(network_nr=49, net_input=[1.1,2.3,0.5], recursion=3)
neat.update_fitness(3, 9)
neat.update_fitness(1, 4)
neat.update_fitness(2, 5)
neat.update_fitness(5, 8)
neat.update_fitness(7, 1)

neat.train(0.2, 0.06, 0.5)



'''
