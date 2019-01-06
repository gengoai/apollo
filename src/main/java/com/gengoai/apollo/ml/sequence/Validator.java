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

package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Example;

import java.io.Serializable;

/**
 * The interface Validator.
 *
 * @author David B. Bracewell
 */
public interface Validator extends Serializable {

   /**
    * The constant ALWAYS_TRUE.
    */
   Validator ALWAYS_TRUE = (Validator) (currentLabel, previousLabel, instance) -> true;

   /**
    * Is valid boolean.
    *
    * @param currentLabel  the current label
    * @param previousLabel the previous label
    * @param instance      the instance
    * @return the boolean
    */
   boolean isValid(String currentLabel, String previousLabel, Example instance);

}//END OF Validator
