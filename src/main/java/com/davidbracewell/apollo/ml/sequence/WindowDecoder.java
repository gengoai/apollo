package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierResult;
import com.davidbracewell.collection.LRUMap;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class WindowDecoder implements Decoder, Serializable {
  private static final long serialVersionUID = 1L;

  @Override
  public LabelingResult decode(SequenceLabeler labeler, Sequence sequence) {
    LabelingResult result = new LabelingResult(sequence.size());
    DecoderState state = null;
    for (ContextualIterator<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
      double[] results = labeler.estimate(
        iterator.next().getFeatures().iterator(),
        labeler.getTransitionFeatures().extract(state)
      );
      ClassifierResult cr = new ClassifierResult(results, labeler.getLabelEncoder());
      result.setLabel(iterator.getIndex(), cr);
      state = new DecoderState(
        state,
        cr.getConfidence(),
        cr.getResult()
      );
    }
    return result;
  }

}// END OF WindowDecoder
