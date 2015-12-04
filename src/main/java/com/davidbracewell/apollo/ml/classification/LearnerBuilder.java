package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.conversion.Cast;
import com.davidbracewell.reflection.BeanMap;
import com.davidbracewell.reflection.Reflect;
import com.davidbracewell.reflection.ReflectionException;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import lombok.NonNull;
import lombok.Setter;
import lombok.experimental.Accessors;

import java.util.HashMap;
import java.util.Map;

/**
 * @author David B. Bracewell
 */
@Accessors(fluent = true)
public class LearnerBuilder {
  @Setter(onParam = @_({@NonNull}))
  private Class<? extends ClassifierLearner> learnerClass;
  @Setter(onParam = @_({@NonNull}))
  private Map<String, Object> parameters = new HashMap<>();


  public LearnerBuilder parameter(String name, Object value) {
    parameters.put(name, value);
    return this;
  }

  public <T extends ClassifierLearner> T build() {
    Preconditions.checkNotNull(learnerClass, "Learner was not set");
    try {
      BeanMap beanMap = new BeanMap(Reflect.onClass(learnerClass).create().get());
      beanMap.putAll(parameters);
      return Cast.as(beanMap.getBean());
    } catch (ReflectionException e) {
      throw Throwables.propagate(e);
    }
  }


}// END OF LearnerBuilder
