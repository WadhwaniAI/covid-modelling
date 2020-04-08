
def create_plots(sol, states_time_matrices, labels, last_time=100, savefig=False, filenames=None):
    ind = sol.t[:last_time]
    
    plt.figure(figsize=(12, 12))
    for i, states_time_matrix in enumerate(states_time_matrices):
        plt.plot(ind, states_time_matrix[2], label=labels[i])
    plt.ylabel('No of people')
    plt.xlabel('Days')
    plt.legend()
    plt.title("Number of undetected people spreading the disease")
    if not savefig:
        plt.show()
    else:
        plt.savefig('../{}'.format(filenames[0]))


    plt.figure(figsize=(12, 12))
    for i, states_time_matrix in enumerate(states_time_matrices):
        plt.plot(ind, states_time_matrix[3] + states_time_matrix[4], label=labels[i])
    plt.ylabel('No of people')
    plt.xlabel('Days')
    plt.legend()
    plt.title("Number of quarantined people")
    if not savefig:
        plt.show()
    else:
        plt.savefig('../{}'.format(filenames[1]))

    plt.figure(figsize=(12, 12))
    for i, states_time_matrix in enumerate(states_time_matrices):
        plt.plot(ind, states_time_matrix[2] + states_time_matrix[4] + states_time_matrix[5] + 
                 states_time_matrix[6] + states_time_matrix[7] + states_time_matrix[8], label=labels[i])
    plt.ylabel('No of people')
    plt.xlabel('Days')
    plt.legend()
    plt.suptitle("Number of active infections as a function of time")
    if not savefig:
        plt.show()
    else:
        plt.savefig('../{}'.format(filenames[2]))

    plt.figure(figsize=(12, 12))
    for i, states_time_matrix in enumerate(states_time_matrices):
        plt.plot(ind, states_time_matrix[2] + states_time_matrix[4] + states_time_matrix[5] + 
                 states_time_matrix[6] + states_time_matrix[7] + states_time_matrix[8] + 
                 states_time_matrix[9] + states_time_matrix[10], label=labels[i])
    plt.ylabel('No of people')
    plt.xlabel('Days')
    plt.legend()
    plt.title("Total No of infections (Active + Recovered + Dead)")
    if not savefig:
        plt.show()
    else:
        plt.savefig('../{}'.format(filenames[3]))

    plt.figure(figsize=(12, 12))
    for i, states_time_matrix in enumerate(states_time_matrices):
        plt.plot(ind, states_time_matrix[7] + states_time_matrix[8], label=labels[i])
    plt.ylabel('No of people')
    plt.xlabel('Days')
    plt.legend()
    plt.title("Number of hospitalisations")
    if not savefig:
        plt.show()
    else:
        plt.savefig('../{}'.format(filenames[4]))


    plt.figure(figsize=(12, 12))
    for i, states_time_matrix in enumerate(states_time_matrices):
        plt.plot(ind, states_time_matrix[10], label=labels[i])
    plt.ylabel('No of people')
    plt.xlabel('Days')
    plt.legend()
    plt.title("Number of deaths")
    plt.show()
    if not savefig:
        plt.show()
    else:
        plt.savefig('../{}'.format(filenames[5]))