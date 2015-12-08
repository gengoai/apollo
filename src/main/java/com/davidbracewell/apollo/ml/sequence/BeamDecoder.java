package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierResult;
import com.davidbracewell.collection.Collect;
import com.google.common.base.Preconditions;
import com.google.common.collect.MinMaxPriorityQueue;
import com.google.common.collect.Ordering;
import lombok.NonNull;

import java.util.LinkedList;
import java.util.List;

/**
 * The type Beam decoder.
 *
 * @author David B. Bracewell
 */
public class BeamDecoder implements Decoder {
  private int beamSize;

  public BeamDecoder() {
    this(3);
  }

  public BeamDecoder(int beamSize) {
    Preconditions.checkArgument(beamSize > 0, "Beam size must be > 0.");
    this.beamSize = beamSize;
  }

  @Override
  public LabelingResult decode(@NonNull SequenceLabeler labeler, @NonNull Sequence sequence) {
    ContextualIterator<Instance> iterator = sequence.iterator();
    MinMaxPriorityQueue<DecoderState> queue = initMatrix(labeler, iterator.next());
    while (iterator.hasNext()) {
      iterator.next();
      queue = fillMatrix(queue, labeler, iterator);
    }
    LabelingResult result = new LabelingResult(sequence.size());
    DecoderState last = queue.remove();
    while (last != null) {
      result.setLabel(last.index, last.tag, last.stateProbability);
      last = last.previousState;
    }
    return result;
  }

  private MinMaxPriorityQueue<DecoderState> initMatrix(SequenceLabeler model, Instance instance) {
    ClassifierResult result = model.estimateInstance(instance);
    MinMaxPriorityQueue<DecoderState> queue = MinMaxPriorityQueue
      .orderedBy(Ordering.natural().reverse())
      .maximumSize(beamSize)
      .create();

    model.getLabelEncoder().values().forEach(label ->
      queue.add(new DecoderState(result.getConfidence(label.toString()), label.toString()))
    );
    return queue;
  }

  private MinMaxPriorityQueue<DecoderState> fillMatrix(MinMaxPriorityQueue<DecoderState> queue, SequenceLabeler model, ContextualIterator<Instance> iterator) {
    List<DecoderState> newStates = new LinkedList<>();
    while (!queue.isEmpty()) {
      DecoderState state = queue.remove();
      Instance instance = iterator.getCurrent();
      Instance ti = Instance.create(
        Collect.union(
          instance.getFeatures(),
          model.getTransitionFeatures().extract(state)
        )
      );
      ClassifierResult result = model.estimateInstance(ti);
      model.getLabelEncoder().values().forEach(label ->
        newStates.add(new DecoderState(state, result.getConfidence(label.toString()), label.toString()))
      );
    }
    queue.addAll(newStates);
    return queue;
  }

  /**
   * Gets beam size.
   *
   * @return the beam size
   */
  public int getBeamSize() {
    return beamSize;
  }

  /**
   * Sets beam size.
   *
   * @param beamSize the beam size
   */
  public void setBeamSize(int beamSize) {
    Preconditions.checkArgument(beamSize > 0, "Beam size must be > 0.");
    this.beamSize = beamSize;
  }

}// END OF BeamDecoder
