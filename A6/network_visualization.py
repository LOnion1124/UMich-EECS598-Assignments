"""
Implements a network visualization in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from network_visualization.py!")


def compute_saliency_maps(X: torch.Tensor, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    # Hint: X.grad.data stores the gradients                                     #
    ##############################################################################
    # Replace "pass" statement with your code
    if X.grad is not None:
        X.grad.zero_()
    # forward
    scores = model(X)                           # shape (N, C)
    # pick correct-class logits and sum so backward can be called once
    correct_scores = scores.gather(1, y.view(-1, 1)).squeeze(1)  # shape (N,)
    # compute gradients of the sum of correct logits wrt X (avoids param grads accumulation)
    grads = torch.autograd.grad(outputs=correct_scores.sum(), inputs=X)[0]  # shape (N,3,H,W)
    saliency = grads.abs().amax(dim=1)          # max over channels -> (N,H,W)
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the pogress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our adversarial attack to the input image, and make it require
    # gradient
    X_adv = X.clone()
    X_adv = X_adv.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate an adversarial attack X_adv that the model will classify    #
    # as the class target_y. You should perform gradient ascent on the score     #
    # of the target class, stopping when the model is fooled.                    #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate an adversarial     #
    # attack in fewer than 100 iterations of gradient ascent.                    #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    # Replace "pass" statement with your code
    for it in range(max_iter):
        scores = model(X_adv) # (1, C)
        
        max_score, cur_y = scores.max(dim=1) # (1, )
        if verbose:
            print(f'Iteration {it}: target score {scores[0, target_y].item(): .3f}, max score {max_score.item(): .3f}')
        if cur_y.item() == target_y:
            break

        if X_adv.grad is not None:
            X_adv.grad.zero_()

        # get scalar target score (tensor) rather than a Python float
        target_score = scores[0, target_y]
        # compute gradient of the target score w.r.t. the input image
        grad = torch.autograd.grad(target_score, X_adv)[0]

        # normalize gradient (add small eps to avoid division by zero)
        grad_norm = torch.norm(grad)
        if grad_norm.item() == 0:
            # cannot make progress
            break
        dX = learning_rate * grad / (grad_norm + 1e-8)

        # apply update without tracking in autograd and keep requires_grad for next iter
        with torch.no_grad():
            X_adv += dX
        X_adv.requires_grad_()
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    # Replace "pass" statement with your code
    scores = model(img) # (1, C)
    reg = l2_reg * (img ** 2).sum()
    loss = scores[0, target_y] - reg

    if img.grad is not None:
        img.grad.zero_()
    
    grad = torch.autograd.grad(loss, img)[0]

    with torch.no_grad():
        img += grad
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return img
