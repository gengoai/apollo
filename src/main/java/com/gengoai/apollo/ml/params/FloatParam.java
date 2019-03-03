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

import java.util.function.Predicate;

/**
 * A specialized Parameter for Float parameters
 *
 * @author David B. Bracewell
 */
public class FloatParam extends Param<Float> {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Float parameter.
    *
    * @param name        the name
    * @param description the description
    */
   public FloatParam(String name, String description) {
      super(name, Float.class, description);
   }

   /**
    * Instantiates a new Float parameter.
    *
    * @param name        the name
    * @param description the description
    * @param validator   the validator
    */
   public FloatParam(String name, String description, Predicate<? super Float> validator) {
      super(name, Float.class, description, validator);
   }

}//END OF FloatParam
