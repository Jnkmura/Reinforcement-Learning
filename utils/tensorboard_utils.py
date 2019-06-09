import tensorflow as tf 
import numpy as np

def add_histogram(writer, tag, values, step, bins=1000):
    """
    Logs the histogram of a list/vector of values.
    From: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """

    counts, bin_edges = np.histogram(values, bins=bins)

    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    bin_edges = bin_edges[1:]
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)