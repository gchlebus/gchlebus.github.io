---
layout: post
title: Dunder Methods
categories: python
---

The term **dunder method** refers to Python special methods that can be implemented in a user defined class. The special methods are always spelled with leading and trailing double underscores, e.g., `__len__` or `__call__`. Important feature of dunder methods is that they are meant to be called by the Python interpreter, not by the user (the only excepton is the `__init__` method).
Smart usage of these methods will make our custom classes more pythonic, thus, easier to use.



Comments:
{% highlight ruby %}
def show
  @widget = Widget(params[:id])
  respond_to do |format|
    format.html # show.html.erb
    format.json { render json: @widget }
  end
end
{% endhighlight %}


https://docs.python.org/3/reference/datamodel.html#special-method-names