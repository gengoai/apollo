package com.davidbracewell.apollo.ml.sequence.linear;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.LibLinearLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceLabeler;
import com.davidbracewell.apollo.ml.sequence.SequenceLabelerLearner;
import com.davidbracewell.conversion.Cast;
import lombok.NonNull;

import java.util.Map;

/**
 * @author David B. Bracewell
 */
public class MEMMLearner extends SequenceLabelerLearner {
   private static final long serialVersionUID = 1L;
   private LibLinearLearner learner = new LibLinearLearner();

   @Override
   protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {
      MEMM model = new MEMM(
                              dataset.getLabelEncoder(),
                              dataset.getFeatureEncoder(),
                              dataset.getPreprocessors().getModelProcessors(),
                              getTransitionFeatures(),
                              getValidator()
      );
      Dataset<Instance> nd = Dataset.classification()
                                    .source(dataset.stream().flatMap(s -> s.asInstances().stream()))
                                    .build();
      dataset.close();
      model.model = Cast.as(learner.train(nd));
      return model;
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
}// END OF MEMMLearner
