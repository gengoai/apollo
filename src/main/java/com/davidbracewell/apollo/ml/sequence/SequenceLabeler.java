package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.apollo.ml.classification.ClassifierResult;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

import java.util.Iterator;

/**
 * The type Sequence labeler.
 *
 * @author David B. Bracewell
 */
public abstract class SequenceLabeler extends Model {
  private static final long serialVersionUID = 1L;
  private final PreprocessorList<Sequence> preprocessors;
  private final TransitionFeatures transitionFeatures;
  private volatile Decoder decoder = new BeamDecoder();

//  private Featurizer featurizer;

  /**
   * Instantiates a new Model.
   *
   * @param labelEncoder       the label encoder
   * @param featureEncoder     the feature encoder
   * @param preprocessors      the preprocessors
   * @param transitionFeatures the transition features
   */
  public SequenceLabeler(@NonNull Encoder labelEncoder, @NonNull Encoder featureEncoder, @NonNull PreprocessorList<Sequence> preprocessors, TransitionFeatures transitionFeatures) {
    super(labelEncoder, featureEncoder);
    this.preprocessors = preprocessors.getModelProcessors();
    this.transitionFeatures = transitionFeatures;
  }

  @Override
  protected void finishTraining() {
    super.finishTraining(); //Call super to freeze encoders
    preprocessors.trimToSize(getFeatureEncoder());
  }

  /**
   * Label labeling result.
   *
   * @param sequence the sequence
   * @return the labeling result
   */
  public LabelingResult label(@NonNull Sequence sequence) {
    return decoder.decode(this, sequence);
  }

  /**
   * Gets decoder.
   *
   * @return the decoder
   */
  public Decoder getDecoder() {
    return decoder;
  }

  /**
   * Sets decoder.
   *
   * @param decoder the decoder
   */
  public void setDecoder(@NonNull Decoder decoder) {
    this.decoder = decoder;
  }

  public abstract double[] estimate(Iterator<Feature> observation, Iterator<String> transitions);

  //  /**
//   * Sets featurizer.
//   *
//   * @param featurizer the featurizer
//   */
//  public void setFeaturizer(Featurizer featurizer) {
//    this.featurizer = featurizer;
//  }


  public TransitionFeatures getTransitionFeatures() {
    return transitionFeatures;
  }
}// END OF SequenceLabeler
