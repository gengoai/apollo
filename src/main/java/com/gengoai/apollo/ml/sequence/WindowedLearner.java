package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.classification.ClassifierLearner;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.io.QuietIO;
import lombok.NonNull;

import java.util.Map;

/**
 * <p>Greedy learner that wraps a {@link ClassifierLearner}.</p>
 *
 * @author David B. Bracewell
 */
public class WindowedLearner extends SequenceLabelerLearner {

   private static final long serialVersionUID = 3783930856969307606L;
   private final ClassifierLearner learner;

   /**
    * Instantiates a new Windowed learner.
    *
    * @param learner the learner
    */
   public WindowedLearner(ClassifierLearner learner) {
      this.learner = learner;
   }

   @Override
   public Object getParameter(String name) {
      return learner.getParameter(name);
   }

   @Override
   public Map<String, ?> getParameters() {
      return learner.getParameters();
   }

   @Override
   public WindowedLearner setParameters(@NonNull Map<String, Object> parameters) {
      learner.setParameters(parameters);
      return this;
   }

   @Override
   public void resetLearnerParameters() {
      learner.reset();
   }

   @Override
   public WindowedLearner setParameter(String name, Object value) {
      learner.setParameter(name, value);
      return this;
   }

   @Override
   protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {
      WindowedLabeler wl = new WindowedLabeler(this);
      Dataset<Instance> nd = Dataset.classification()
                                    .source(dataset.stream()
                                                   .flatMap(sequence -> getTransitionFeatures().toInstances(sequence)
                                                                                               .stream()));
      QuietIO.closeQuietly(dataset);
      wl.classifier = learner.train(nd);
      wl.encoderPair = wl.classifier.getEncoderPair();
      return wl;
   }

}// END OF WindowedLearner
