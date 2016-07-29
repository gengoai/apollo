package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import lombok.NonNull;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class WindowedLearner extends SequenceLabelerLearner {

  private final ClassifierLearner learner;

  public WindowedLearner(ClassifierLearner learner) {
    this.learner = learner;
  }

  @Override
  protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {
    WindowedLabeler wl = new WindowedLabeler(
      dataset.getLabelEncoder(),
      dataset.getFeatureEncoder(),
      dataset.getPreprocessors(),
      getTransitionFeatures(),
      getValidator()
    );

    Dataset<Instance> nd = Dataset.classification()
      .source(dataset.stream().flatMap(sequence -> getTransitionFeatures().toInstances(sequence)))
      .build();

    dataset.close();
    wl.classifier = learner.train(nd);
    wl.encoderPair = wl.classifier.getEncoderPair();
    return wl;
  }

  @Override
  public void reset() {
    learner.reset();
  }




  @Override
  public Map<String, ?> getParameters() {
    return learner.getParameters();
  }

  @Override
  public void setParameters(@NonNull Map<String, Object> parameters) {
    learner.setParameters(parameters);
  }

  @Override
  public void setParameter(String name, Object value) {
    learner.setParameter(name, value);
  }

  @Override
  public Object getParameter(String name) {
    return learner.getParameter(name);
  }

}// END OF WindowedLearner
