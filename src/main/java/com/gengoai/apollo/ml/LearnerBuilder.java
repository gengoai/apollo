package com.gengoai.apollo.ml;

import com.gengoai.guava.common.base.Preconditions;
import com.gengoai.guava.common.base.Throwables;
import com.gengoai.mango.reflection.Reflect;
import com.gengoai.mango.reflection.ReflectionException;
import lombok.NonNull;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * <p>Builder for Learners</p>
 *
 * @param <T> the example type parameter for what type of examples the learner accepts
 * @param <M> the model type parameter for what type of models the learner trains
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public class LearnerBuilder<T extends Example, M extends Model> implements Serializable {
   private static final long serialVersionUID = 1L;
   private Class<? extends Learner<T, M>> learnerClass;
   private Map<String, Object> parameters = new HashMap<>();

   /**
    * Builds a learner using the configured learner class and parameters
    *
    * @param <R> the learner type parameter
    * @return the learner
    */
   public <R extends Learner<T, M>> R build() {
      Preconditions.checkNotNull(learnerClass, "Learner was not set");
      try {
         R learner = Reflect.onClass(learnerClass).create().get();
         learner.setParameters(parameters);
         return learner;
      } catch (ReflectionException e) {
         throw Throwables.propagate(e);
      }
   }

   /**
    * Sets the given parameter to the given value
    *
    * @param name  the name of the parameter to set
    * @param value the value to the set the parameter to
    * @return the learner builder
    */
   public LearnerBuilder<T, M> parameter(String name, Object value) {
      parameters.put(name, value);
      return this;
   }

   /**
    * Sets the class of learner that we are building.
    *
    * @param learnerClass the learner class
    */
   public LearnerBuilder<T, M> learnerClass(@NonNull Class<? extends Learner<T, M>> learnerClass) {
      this.learnerClass = learnerClass;
      return this;
   }

   /**
    * Creates a supplier from this builder that calls build on each request.
    *
    * @param <R> the learner type parameter
    * @return the supplier
    */
   public <R extends Learner<T, M>> Supplier<R> supplier() {
      return this::build;
   }


}// END OF LearnerBuilder
