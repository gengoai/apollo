package com.davidbracewell.apollo.ml.sequence.decoder;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.sequence.ContextualIterator;
import com.davidbracewell.apollo.ml.sequence.Labeling;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceLabeler;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class WindowDecoder implements Decoder, Serializable {
  private static final long serialVersionUID = 1L;

  @Override
  public Labeling decode(SequenceLabeler labeler, Sequence sequence) {
    Labeling result = new Labeling(sequence.size());
    DecoderState state = null;
    String previousLabel = null;

    for (ContextualIterator<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
      double[] results = labeler.estimate(
        iterator.next().getFeatures().iterator(),
        labeler.getTransitionFeatures().extract(state)
      );

      double max = Double.NEGATIVE_INFINITY;
      String label = null;
      for (int i = 0; i < results.length; i++) {
        String tL = labeler.getEncoderPair().decodeLabel(i).toString();
        if (results[i] > max && labeler.getValidator().isValid(tL, previousLabel, iterator.getCurrent())) {
          max = results[i];
          label = tL;
        }
      }
      if (max == Double.NEGATIVE_INFINITY) {
        for (int i = 0; i < results.length; i++) {
          String tL = labeler.getEncoderPair().decodeLabel(i).toString();
          if (results[i] > max) {
            max = results[i];
            label = tL;
          }
        }
      }
      previousLabel = label;
      result.setLabel(iterator.getIndex(), label, max);
      state = new DecoderState(
        state,
        max,
        label
      );
    }
    return result;
  }

}// END OF WindowDecoder
