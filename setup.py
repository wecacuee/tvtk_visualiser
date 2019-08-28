from setuptools import setup
setup(name="tvtk_visualiser",
      description="A lightweight wrapper over tvtk for visualization without using mayavi.",
      author="Julian Ryde and Vikas Dhiman",
      url='https://github.com/wecacuee/tvtk_visualiser',
      version="1.0.0",
      author_email='wecacuee@github.com',
      long_description=open('README.md', encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      license='MIT',
      classifiers=(
          'Development Status :: 3 - Alpha',
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ),
      python_requires='>=3.5',
      py_modules=["tvtk_visualiser", "tvtk_utils"],
      install_requires=["numpy", "matplotlib", "mayavi", "PyQt5"])
