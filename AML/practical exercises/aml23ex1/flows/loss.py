def likelihood(X_train, model, device):
    ##########################################################
    # YOUR CODE HERE
    log_probs = model.log_prob(X_train.to(device))
    loss = -log_probs.mean(0)
    ##########################################################

    return loss
