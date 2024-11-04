import torch
import torch.nn as nn


def rnn_output_test(
    i2dl_rnn,
    pytorch_rnn,
    x,
    val=0.3,
    tol=1e-9
):

    for p in pytorch_rnn.parameters():
        nn.init.constant_(p, val=val)
    for p in i2dl_rnn.parameters():
        nn.init.constant_(p, val=val)

    output_pytorch, h_pytorch = pytorch_rnn(x)
    output_i2dl, h_i2dl = i2dl_rnn(x)

    if isinstance(h_pytorch, tuple):
        assert isinstance(h_i2dl, tuple) and len(h_pytorch) == len(h_i2dl), \
            'Different rnn types {} vs {}!'.format(
                i2dl_rnn.__class__.__name__, pytorch_rnn.__class__.__name__
            )
        h_pytorch, c_pytorch = h_pytorch
        h_i2dl, c_i2dl = h_i2dl
    else:
        c_pytorch, c_i2dl = None, None

    # Outputs must have the same shapes
    passed = True
    if output_pytorch.data.shape == output_i2dl.data.shape:
        print('Output shape test passed :), {} == {}'.format(
            output_pytorch.data.shape, output_i2dl.data.shape
        ))
    else:
        print('Output shape test failed :(, {} != {}'.format(
            output_pytorch.shape, output_i2dl.shape
        ))
        passed = False
    if h_pytorch.shape == h_i2dl.shape:
        print('Hidden shape test passed :), {} == {}'.format(
            h_pytorch.shape, h_i2dl.shape
        ))
    else:
        print('Hidden shape test failed :(, {} != {}'.format(
            h_pytorch.shape, h_i2dl.shape
        ))
        passed = False

    for output, name in zip(
        [(output_i2dl, output_pytorch), (h_i2dl, h_pytorch), (c_i2dl, c_pytorch)],
        ['h_seq', 'h', 'c']
    ):
        if output[0] is None or output[1] is None:
            continue

        if not passed:
            print('Your model has some shape mismatches, check your implementation!')
        else:
            # The difference of outputs should be 0!!
            diff = torch.sum((output[0].data - output[1].data)**2)
            print("\nDifference between pytorch and your RNN implementation for '{}': {:.2f}".format(
                name, diff.item()
            ))
            if diff.item() < tol:
                print("Cool, you implemented a correct model.")
            else:
                print("Upps! There is something wrong in your model. Try again!")
                passed = False
                break

    return passed


def embedding_output_test(
    i2dl_embedding,
    pytorch_embedding,
    x,
    val=0.3,
    tol=1e-9
):
    i2dl_embedding.weight.data.copy_(pytorch_embedding.weight.data)

    i2dl_output = i2dl_embedding(x)
    torch_output = pytorch_embedding(x)
    passed = True
    if i2dl_output.shape != torch_output.shape:
        passed = False
        print('Output shapes are mismatched! {} vs {}'.format(
            i2dl_output.shape, torch_output.shape
        ))

    if passed:
        diff = (i2dl_output - torch_output).pow(2).sum().sqrt().item()
        print('Difference between outputs: {}'.format(diff))

        if diff < 1e-9:
            print('Test passed :)!')
        else:
            print('Test failed, check you implementation :(!')
            passed = False

    return passed


def classifier_test(classifier, num_embeddings):
    # Define some constants
    seq_len=10
    batch_size=3
    
    # Create a random sequence
    x = torch.randint(0, num_embeddings-1, (seq_len, batch_size))

    # Test the output format
    y = classifier(x)
    passed = True
    if not torch.logical_and((y <= 1), (y >= 0)).all():
        print('Your model does not output probabilities between 0 and 1!')
        passed = False
    if y.shape != (batch_size, ):
        print('Your model does not produce a 1-D output of shape (batch_size, )')
        passed = False

    # Test varying batch sizes
    assert seq_len-batch_size > 0, "Seq len must be bigger than batch size"
    lengths = torch.tensor([seq_len-i for i in range(batch_size)]).long()
    batched_outputs = classifier(x, lengths)
    regular_outputs = torch.stack([
        classifier(x[:lengths[i], i].unsqueeze(1))
        for i in range(lengths.numel())
    ]).squeeze()

    if batched_outputs.shape != regular_outputs.shape:
        print('Output with lengths {} produced wrong size argument {} vs {}'.format(
            lengths.tolist(), batched_outputs.shape, regular_outputs.shape
        ))
        print('Make sure you handle lengths argument properly in your classifier!')
        passed = False

    diff = torch.norm(batched_outputs - regular_outputs)
    if diff > 1e-9:
        print('Output with lengths {} has a large error: {}'.format(lengths.tolist(), diff))
        print('Make sure you handle lengths argument properly in your classifier!') 
        passed = False

    # Log the final result
    if passed:
        print('All output tests are passed :)!')
    else:
        print('Some output tests are failed :(!')
    return passed


def parameter_test(model):
    total = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: {}'.format(total))
    if total < 2 * 1e6:
        print('Your model is sufficiently small :)')
        return True
    else:
        print('Your model is too large :(! Shrink its size!')
        return False
