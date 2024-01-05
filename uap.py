def generate_uap(model, data_loader, epsilon=0.01, max_iterations=10):
    """
    Generates a universal adversarial perturbation.
    :param model: The neural network model.
    :param data_loader: DataLoader for the dataset.
    :param epsilon: The perturbation magnitude.
    :param max_iterations: Number of iterations to refine the perturbation.
    :return: A universal adversarial perturbation.
    """
    uap = torch.zeros_like(next(iter(data_loader))[0][0])  # Assuming the first batch and the first image shape
    model.eval()

    for _ in range(max_iterations):
        for images, _ in data_loader:
            images = images.cuda()
            images.requires_grad = True

            preds = model(images)
            loss = -1 * torch.nn.functional.cross_entropy(preds, preds.max(1)[1])  # Maximize the loss
            model.zero_grad()
            loss.backward()

            uap -= epsilon * images.grad.sign()
            uap = torch.clamp(uap, -epsilon, epsilon)  # Ensure uap is within bounds

    return uap
