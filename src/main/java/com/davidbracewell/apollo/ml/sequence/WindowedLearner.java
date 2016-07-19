package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import lombok.NonNull;

import java.util.Map;

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
      .source(dataset.stream().flatMap(Sequence::asInstances))
      .build();
    dataset.close();
    wl.classifier = learner.train(nd);
    return wl;
  }

  @Override
  public void reset() {

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
