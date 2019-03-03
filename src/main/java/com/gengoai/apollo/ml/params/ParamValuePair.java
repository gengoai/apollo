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

import java.io.Serializable;

/**
 * Encapsulates a {@link Param} and validated value. Should only be created via {@link Param#set(Object)}
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public class ParamValuePair<T> implements Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The Param.
    */
   public final Param<T> param;
   /**
    * The Value.
    */
   public final T value;

   /**
    * Instantiates a new Param value pair.
    *
    * @param param the param
    * @param value the value
    */
   protected ParamValuePair(Param<T> param, T value) {
      this.param = param;
      this.value = value;
   }

}//END OF ParamValuePair
