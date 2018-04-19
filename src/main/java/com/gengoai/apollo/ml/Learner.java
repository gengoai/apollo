package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.classification.Classifier;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.encoder.EncoderPair;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.mango.conversion.Cast;
import com.gengoai.mango.reflection.BeanMap;
import com.gengoai.mango.reflection.Ignore;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Map;

/**
 * <p>Base class for methods to learn (train) a model of a specific type.</p>
 *
 * @param <T> the example type parameter
 * @param <M> the model type parameter
 * @author David B. Bracewell
 */
public abstract class Learner<T extends Example, M extends Model> implements Serializable {

   private static final long serialVersionUID = 605756060816072642L;
   @Getter
   private EncoderPair encoderPair;
   @Getter
   private PreprocessorList<T> preprocessors;

   /**
    * Creates a builder for constructing Classification learners
    *
    * @return the learner builder
    */
   public static LearnerBuilder<Instance, Classifier> classification() {
      return new LearnerBuilder<>();
   }

//   /**
//    * Creates a builder for constructing clusterers
//    *
//    * @return the learner builder
//    */
//   public static <T extends Clustering> LearnerBuilder<Instance, T> clustering() {
//      return new LearnerBuilder<>();
//   }
//
//   /**
//    * Creates a builder for constructing Embedding learners
//    *
//    * @return the learner builder
//    */
//   public static LearnerBuilder<Sequence, Embedding> embedding() {
//      return new LearnerBuilder<>();
//   }
//
//   /**
//    * Creates a builder for constructing Regression learners
//    *
//    * @return the learner builder
//    */
//   public static LearnerBuilder<Instance, Regression> regression() {
//      return new LearnerBuilder<>();
//   }
//
//   /**
//    * Creates a builder for constructing Sequence learners
//    *
//    * @return the learner builder
//    */
//   public static LearnerBuilder<Sequence, SequenceLabeler> sequence() {
//      return new LearnerBuilder<>();
//   }

   /**
    * Gets the value of the given parameter.
    *
    * @param name the name of the parameter
    * @return the parameter's value or null
    */
   @Ignore
   public Object getParameter(String name) {
      return new BeanMap(this).get(name);
   }

   /**
    * Gets the  parameters that can be set/retrieved on this learner.
    *
    * @return the parameters
    */
   @Ignore
   public Map<String, ?> getParameters() {
      return new BeanMap(this);
   }

   /**
    * Sets the parameters of this learner using the supplied parameter map.
    *
    * @param parameters the parameters
    */
   @Ignore
   public Learner<T, M> setParameters(@NonNull Map<String, Object> parameters) {
      new BeanMap(this).putAll(parameters);
      return Cast.as(this);
   }

   /**
    * Resets any saved state in the learner.
    */
   public final void reset() {
      this.encoderPair = null;
      this.preprocessors = null;
      resetLearnerParameters();
   }

   protected abstract void resetLearnerParameters();

   /**
    * Sets the value of the given parameter.
    *
    * @param name  the name of the parameter
    * @param value the value to set the parameter to
    */
   @Ignore
   public Learner<T, M> setParameter(String name, Object value) {
      new BeanMap(this).put(name, value);
      return Cast.as(this);
   }


   /**
    * Trains a model using the given dataset.
    *
    * @param dataset the dataset to use for training
    * @return the trained model
    */
   public M train(@NonNull Dataset<T> dataset) {
      dataset.encode();
      this.preprocessors = dataset.getPreprocessors();
      this.encoderPair = dataset.getEncoderPair();
      M model = trainImpl(dataset);
      model.finishTraining();
      return model;
   }

   /**
    * Actual training implementation to be defined by individual learners
    *
    * @param dataset the dataset to use for training
    * @return the trained model
    */
   protected abstract M trainImpl(Dataset<T> dataset);

   public void update(EncoderPair encoderPair, PreprocessorList<T> preprocessors) {
      this.encoderPair = encoderPair;
      this.preprocessors = preprocessors;
   }
}// END OF Learner
