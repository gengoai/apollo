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
 */

package com.gengoai.apollo.ml;

import lombok.NonNull;
import lombok.Value;

import java.io.Serializable;

/**
 * Simple container for an object and its associated label
 *
 * @author David B. Bracewell
 */
@Value
public class LabeledDatum<T> implements Serializable {
   private static final long serialVersionUID = 1L;
   private Object label;
   private T data;

   /**
    * Convenience method for creating a labeled data point
    *
    * @param <R>   the data type parameter
    * @param label the label
    * @param data  the data
    * @return the labeled data
    */
   public static <R> LabeledDatum<R> of(Object label, @NonNull R data) {
      return new LabeledDatum<>(label, data);
   }

}//END OF LabeledDatum
