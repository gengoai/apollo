/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

package com.gengoai.apollo.ml.params;

import com.gengoai.Validation;

import java.io.Serializable;
import java.util.function.Predicate;

/**
 * <p>A named parameter used for fitting a model to a dataset. Defines the name and type of the parameter and
 * provides a textual description and method to validate potential values.</p>
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public class Param<T> implements Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The Name.
    */
   public final String name;
   /**
    * The Class information for the type of the value
    */
   public final Class<T> valueType;
   /**
    * Describes the parameter for digest by end users.
    */
   public final String description;
   /**
    * Validation method for potential parameter values
    */
   public final Predicate<? super T> validator;



   /**
    * Instantiates a new Parameter.
    *
    * @param name        the name of the parameter
    * @param valueType   the type of the value it accepts
    * @param description the description of the parameter
    * @param validator   the validation method used to check potential values
    */
   public Param(String name, Class<T> valueType, String description, Predicate<? super T> validator) {
      this.name = name;
      this.valueType = valueType;
      this.description = description;
      this.validator = validator;
   }

   /**
    * Instantiates a new Parameter that accepts any value of the given valueType.
    *
    * @param name        the name of the parameter
    * @param valueType   the type of the value it accepts
    * @param description the description of the parameter
    */
   public Param(String name, Class<T> valueType, String description) {
      this.name = name;
      this.valueType = valueType;
      this.description = description;
      this.validator = o -> true;
   }


   /**
    * Creates a {@link ParamValuePair} combining this parameter and the given value. The value is validated using the
    * validation method defined on the parameter.
    *
    * @param value the value to set
    * @return the param value pair
    * @throws IllegalArgumentException If the value is not valid
    */
   public ParamValuePair<T> set(T value) {
      Validation.validateArg(value, validator, false);
      return new ParamValuePair<>(this, value);
   }

   @Override
   public String toString() {
      return "Param{" +
                "name='" + name + '\'' +
                ", type=" + valueType +
                ", description='" + description + '\'' +
                '}';
   }
}//END OF Param
