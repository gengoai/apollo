package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.regression.Regression;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceLabeler;
import com.davidbracewell.reflection.BeanMap;
import com.davidbracewell.reflection.Ignore;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Map;

/**
 * The type Learner.
 *
 * @param <T> the type parameter
 * @param <M> the type parameter
 * @author David B. Bracewell
 */
public abstract class Learner<T extends Example, M extends Model> implements Serializable {

   private static final long serialVersionUID = 605756060816072642L;

   /**
    * Builder learner builder.
    *
    * @param <T> the type parameter
    * @param <M> the type parameter
    * @return the learner builder
    */
   public static <T extends Example, M extends Model> LearnerBuilder<T, M> builder() {
      return new LearnerBuilder<>();
   }

   /**
    * Classification learner builder.
    *
    * @return the learner builder
    */
   public static LearnerBuilder<Instance, Classifier> classification() {
      return new LearnerBuilder<>();
   }

   /**
    * Regression learner builder.
    *
    * @return the learner builder
    */
   public static LearnerBuilder<Instance, Regression> regression() {
      return new LearnerBuilder<>();
   }

   /**
    * sequence learner builder.
    *
    * @return the learner builder
    */
   public static LearnerBuilder<Sequence, SequenceLabeler> sequence() {
      return new LearnerBuilder<>();
   }

   /**
    * Gets parameter.
    *
    * @param name the name
    * @return the parameter
    */
   @Ignore
   public Object getParameter(String name) {
      return new BeanMap(this).get(name);
   }

   /**
    * Gets parameters.
    *
    * @return the parameters
    */
   @Ignore
   public Map<String, ?> getParameters() {
      return new BeanMap(this);
   }

   /**
    * Sets parameters.
    *
    * @param parameters the parameters
    */
   @Ignore
   public void setParameters(@NonNull Map<String, Object> parameters) {
      new BeanMap(this).putAll(parameters);
   }

   /**
    * Reset.
    */
   public abstract void reset();

   /**
    * Sets parameter.
    *
    * @param name  the name
    * @param value the value
    */
   @Ignore
   public void setParameter(String name, Object value) {
      new BeanMap(this).put(name, value);
   }

   /**
    * Train classifier.
    *
    * @param dataset the dataset
    * @return the classifier
    */
   public M train(@NonNull Dataset<T> dataset) {
      dataset.encode();
      M model = trainImpl(dataset);
      model.finishTraining();
      return model;
   }

   /**
    * Train m.
    *
    * @param dataset the dataset
    * @return the m
    */
   protected abstract M trainImpl(Dataset<T> dataset);


}// END OF Learner
