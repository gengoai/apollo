package com.davidbracewell.apollo.ml;

import com.davidbracewell.reflection.Reflect;
import com.davidbracewell.reflection.ReflectionException;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import lombok.NonNull;
import lombok.Setter;
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
   @Setter(onParam = @_({@NonNull}))
   private Class<? extends Learner<T, M>> learnerClass;
   private Map<String, Object> parameters = new HashMap<>();


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
    * Creates a supplier from this builder that calls build on each request.
    *
    * @param <R> the learner type parameter
    * @return the supplier
    */
   public <R extends Learner<T, M>> Supplier<R> supplier() {
      return this::build;
   }


}// END OF LearnerBuilder
